import torch.nn.functional
from transformers import AutoTokenizer, RobertaForMaskedLM, RobertaModel, get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_loader import *
from model.metric import load_metric
# import wandb
from model.model_utils.tokenizer_utils import *
from datetime import datetime
import os
from shutil import copyfile
from model.models import *
import random
import numpy as np


def adpxt_runner(config):
    for job_idx, name in enumerate(config['job_name']['name']):
        SEED = config['seed'][job_idx]
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # wandb.init(project=config['project_name'], reinit=True)
        now = datetime.now()
        save_dir = f'./saved/models/{config["project_name"]}/{name}/'
        name = name + now.strftime('-%Y-%m-%d-%H%M%S')
        os.makedirs(save_dir)
        copyfile(config['config_name'], save_dir + config['config_name'])
        # wandb.run.name = name
        # wandb.run.save()

        data_path = config['data_loader']['args']['data_dir']
        domains = config['data_loader']['args']['source_domains']
        target_domain = config['data_loader']['args']['target_domain']

        plm_type = config['arch']['plm']
        tokenizer_type = config['arch']['tokenizer']

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        template = config['template']
        # 데이터 셋
        if config['data_loader']['dataset'] == 'mnli':
            if config['data_loader']['args']['training_setting'] == 'fewshot':
                train_datasets = MNLIDataset(data_path, domains, target_domain, template, tokenizer, 'train_fewshot',
                                             SEED, config['shots'])
            else:
                train_datasets = MNLIDataset(data_path, domains, target_domain, template, tokenizer, 'train')
            dev_datasets = MNLIDataset(data_path, domains, target_domain, template, tokenizer, 'dev')
            test_datasets = MNLIDataset(data_path, domains, target_domain, template, tokenizer, 'test')


        else:
            if config['data_loader']['args']['training_setting'] == 'fewshot':
                train_datasets = AmazonDataset(data_path, domains, target_domain, template, tokenizer, 'train_fewshot',
                                               SEED, config['shots'])
            else:
                train_datasets = AmazonDataset(data_path, domains, target_domain, template, tokenizer, 'train')
            dev_datasets = AmazonDataset(data_path, domains, target_domain, template, tokenizer, 'dev')
            test_datasets = AmazonDataset(data_path, domains, target_domain, template, tokenizer, 'test')

        train_dataloader = DataLoader(train_datasets,
                                      shuffle=config['data_loader']['args']['shuffle'],
                                      batch_size=config['data_loader']['args']['batch_size'][job_idx],
                                      collate_fn=batch_collate_for_roberta
                                      )
        eval_dataloader = DataLoader(dev_datasets,
                                     shuffle=False,
                                     batch_size=config['data_loader']['args']['dev_batch_size'],
                                     collate_fn=batch_collate_for_roberta)
        test_dataloader = DataLoader(test_datasets,
                                     shuffle=False,
                                     batch_size=config['data_loader']['args']['dev_batch_size'],
                                     collate_fn=batch_collate_for_roberta)
        plm_model = RobertaForMaskedLM.from_pretrained(plm_type, output_hidden_states=True)

        # for freezing model
        if config['freeze_plm']:
            for para in plm_model.parameters():
                para.requires_grad = False

        plm_param = 0
        print(f'{name} model training')
        for p_name, param in plm_model.named_parameters():
            plm_param += param.numel()
        print(f'plm_param: {plm_param}')

        model = AdPXT(plm_model, tokenizer,
                      global_length=config['hyperparam']['global_length'][job_idx],
                      context_length=config['hyperparam']['context_length'][job_idx],
                      model_lr=config['optimizer']['args']['model_lr'][job_idx],
                      w_decay=config['optimizer']['args']['weight_decay'],
                      drop=config['hyperparam']['drop_out'],
                      task=config['task'],
                      adversarial=config['adversarial'],
                      num_domains=len(domains),
                      alpha=config['hyperparam']['alpha']
                      )
        print(f' trainable params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

        num_epochs = config['trainer']['epochs']
        num_training_steps = num_epochs * len(train_dataloader)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        model_optimizer = model.model_optimizer
        model_scheduler = get_scheduler(config['lr_scheduler']['type'], optimizer=model_optimizer,
                                        num_warmup_steps=num_training_steps * config['lr_scheduler']['args']['warm_up'],
                                        num_training_steps=num_training_steps)
        if config['adversarial']:
            if config['task'] == 'nli':
                label_aware_domain_classifiers = [
                    Domain_classifier(plm_model.config.hidden_size,
                                      len(domains),
                                      drop=config['hyperparam']['drop_out']).to(device)
                    for _ in range(3)
                ]
                domain_parameter = [
                    *[{"params": label_aware_domain_classifiers[class_idx].parameters() for class_idx in range(3)}]
                ]

            elif config['task'] == 'sa':
                label_aware_domain_classifiers = [
                    Domain_classifier(plm_model.config.hidden_size,
                                      len(domains),
                                      drop=config['hyperparam']['drop_out']).to(device)
                    for _ in range(2)
                ]
                domain_parameter = [
                    *[{"params": label_aware_domain_classifiers[class_idx].parameters() for class_idx in range(2)}]
                ]

            domain_optimizer = AdamW(domain_parameter,
                                     lr=config['optimizer']['args']['disc_lr'][job_idx],
                                     weight_decay=config['optimizer']['args']['weight_decay'])
            domain_scheduler = get_scheduler(config['lr_scheduler']['type'], optimizer=domain_optimizer,
                                             num_warmup_steps=num_training_steps * config['lr_scheduler']['args'][
                                                 'warm_up'],
                                             num_training_steps=num_training_steps)

        best_acc = 0
        CE = torch.nn.CrossEntropyLoss(reduction='mean')
        BCE = torch.nn.BCEWithLogitsLoss(reduction='mean')
        ckpt_path = os.path.join(save_dir, f'model_best_{0}ep.pth')


        for epoch in range(num_epochs):
            # train
            model.train()
            if config['adversarial']:
                for classifier in label_aware_domain_classifiers:
                    classifier.train()
            pbar = tqdm(train_dataloader, desc="Train epoch {}".format(epoch + 1))

            cls_loss_train = 0.
            acc_train = 0.
            disc_loss_train = 0.
            for step, (input_ids, attention_mask, label, domain_idx) in enumerate(pbar):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                label = label.to(device)
                outputs = model(input_ids, attention_mask)
                task_logit = outputs.task_logit

                if config['adversarial']:
                    if config['task'] == 'sa':
                        cls_loss = BCE(task_logit, label.unsqueeze(1).float())
                        acc = load_metric('bce_acc', task_logit, label.unsqueeze(1))
                        cls_loss_train += cls_loss.item()
                        acc_train += acc
                    elif config['task'] == 'nli':
                        cls_loss = CE(task_logit, label)
                        cls_loss_train += cls_loss.item()
                        acc = load_metric('accuracy', task_logit, label)
                        acc_train += acc

                    domain_label = domain_idx.to(device)
                    reversal_feature = outputs.reversal_feature

                    domain_loss_batch = 0
                    for batch_idx, l in enumerate(label):
                        domain_logit = label_aware_domain_classifiers[l](reversal_feature[batch_idx].unsqueeze(0))
                        domain_loss = CE(domain_logit, domain_label[batch_idx].reshape(-1))
                        domain_loss_batch = domain_loss_batch + domain_loss

                    domain_loss_batch = domain_loss_batch / len(label)
                    disc_loss_train += domain_loss_batch.item()
                    pbar.set_postfix({'acc': acc,
                                      'cls_loss': cls_loss.item(),
                                      'disc_loss': domain_loss_batch.item()})

                    total_loss = cls_loss + domain_loss_batch
                    model_optimizer.zero_grad()
                    domain_optimizer.zero_grad()
                    total_loss.backward()
                    model_optimizer.step()
                    domain_optimizer.step()
                    model_scheduler.step()
                    domain_scheduler.step()

                else:
                    if config['task'] == 'sa':
                        cls_loss = BCE(task_logit, label.unsqueeze(1).float())
                        acc = load_metric('bce_acc', task_logit, label.unsqueeze(1))
                        cls_loss_train += cls_loss.item()
                        acc_train += acc
                    elif config['task'] == 'nli':
                        cls_loss = CE(task_logit, label)
                        cls_loss_train += cls_loss.item()
                        acc = load_metric('accuracy', task_logit, label)
                        acc_train += acc

                    pbar.set_postfix({'acc': acc,
                                      'cls_loss': cls_loss.item()})

                    model_optimizer.zero_grad()
                    cls_loss.backward()
                    model_optimizer.step()
                    model_scheduler.step()

            print(
                f'cls loss: {cls_loss_train / step},'
                f'disc loss: {disc_loss_train / step},'
                f'acc: {acc_train / step}')
            print('###############################################################')
            # wandb.log({"cls/train": cls_loss_train / step, 'epoch': epoch})
            # wandb.log({"disc/train": disc_loss_train / step, 'epoch': epoch})
            # wandb.log({"acc/train": acc_train / step, 'epoch': epoch})
            # wandb.log({"lr": model_scheduler.get_last_lr()[-1], 'epoch': epoch})

            # validation
            model.eval()
            if config['adversarial']:
                for classifier in label_aware_domain_classifiers:
                    classifier.eval()
            print('evaluation')
            with torch.no_grad():
                cls_loss_dev = 0
                disc_loss_dev = 0
                acc_dev = 0
                for step, (input_ids, attention_mask, label, domain_idx) in enumerate(eval_dataloader):
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    label = label.to(device)

                    outputs = model(input_ids, attention_mask)
                    task_logit = outputs.task_logit

                    if config['adversarial']:
                        if config['task'] == 'sa':
                            cls_loss = BCE(task_logit, label.unsqueeze(1).float())
                            acc = load_metric('bce_acc', task_logit, label.unsqueeze(1))
                            cls_loss_dev += cls_loss.item()
                            acc_dev += acc
                        elif config['task'] == 'nli':
                            cls_loss = CE(task_logit, label)
                            cls_loss_dev += cls_loss.item()
                            acc = load_metric('accuracy', task_logit, label)
                            acc_dev += acc

                        domain_label = domain_idx.to(device)
                        reversal_feature = outputs.reversal_feature

                        domain_loss_batch = 0
                        for batch_idx, l in enumerate(label):
                            domain_logit = label_aware_domain_classifiers[l](reversal_feature[batch_idx].unsqueeze(0))
                            domain_loss = CE(domain_logit, domain_label[batch_idx].reshape(-1))
                            domain_loss_batch = domain_loss_batch + domain_loss

                        domain_loss_batch = domain_loss_batch / len(label)
                        disc_loss_dev += domain_loss_batch.item()

                    else:
                        if config['task'] == 'sa':
                            cls_loss = BCE(task_logit, label.unsqueeze(1).float())
                            acc = load_metric('bce_acc', task_logit, label.unsqueeze(1))
                            cls_loss_dev += cls_loss.item()
                            acc_dev += acc
                        elif config['task'] == 'nli':
                            cls_loss = CE(task_logit, label)
                            cls_loss_dev += cls_loss.item()
                            acc = load_metric('accuracy', task_logit, label)
                            acc_dev += acc

                # wandb.log({"cls/val": cls_loss_dev / step, 'epoch': epoch})
                # wandb.log({"disc/val": disc_loss_dev / step, 'epoch': epoch})
                # wandb.log({"acc/val": acc_dev / step, 'epoch': epoch})
                print(
                    f'dev cls loss: {cls_loss_dev / step}, dev acc: {acc_dev / step}')
                print('#########################################')

            if acc_dev / step >= best_acc:
                if os.path.isfile(ckpt_path):
                    os.remove(ckpt_path)
                best_acc = acc_dev / step
                torch.save(model, f'{save_dir}model_best_{epoch + 1}ep.pth')
                ckpt_path = os.path.join(save_dir, f'model_best_{epoch + 1}ep.pth')

            # test
            # model.eval()
            # print('test')
            # with torch.no_grad():
            #     temp_acc = 0
            #     for step, (input_ids, attention_mask, label, _) in enumerate(test_dataloader):
            #         input_ids = input_ids.to(device)
            #         attention_mask = attention_mask.to(device)
            #         label = label.to(device)
            #         outputs = model(input_ids, attention_mask)
            #         task_logit = outputs.task_logit
            #         if config['task'] == 'sa':
            #             acc = load_metric('bce_acc', task_logit, label.unsqueeze(1))
            #             temp_acc += acc
            #         elif config['task'] == 'nli':
            #             acc = load_metric('accuracy', task_logit, label)
            #             temp_acc += acc
            #
            #     # wandb.log({"acc/test": temp_acc / step, 'epoch': epoch})
            #     print(f'test acc: {temp_acc / step}')
            #     print('#########################################')

