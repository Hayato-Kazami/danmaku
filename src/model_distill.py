from matplotlib import pyplot as plt
import torch
from utils import build_dataloader, set_seed
from bert_classifier import TMFBertClassifier
from student_model import BiLSTM
from torch.nn import CrossEntropyLoss, KLDivLoss
from config import Config
from train import model2eval
from tqdm import tqdm

conf = Config()


def model2distill(train_loader, dev_loader, teacher_model, student_model, device):
    # 软损失（KL 散度）+ 硬损失（交叉熵）
    soft_loss = KLDivLoss(reduction='batchmean')
    hard_loss = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=conf.student_lr)

    T = conf.temperature
    alpha = conf.alpha

    total_iters = 0
    loss_list = []
    iter_list = []
    best_f1 = 0.

    for epoch in range(conf.epochs):
        teacher_model.eval()
        student_model.train()
        total_loss_epoch = 0

        for i, (input_ids, attention_mask, labels) in enumerate(
                tqdm(train_loader, total=len(train_loader), desc=f"模型蒸馏中")):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # 老师模型输出（不参与梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids, attention_mask)

            # 学生模型输出
            student_logits = student_model(input_ids, attention_mask)

            # 软损失 + 硬损失
            soft_losses = soft_loss(torch.log_softmax(student_logits / T, dim=-1),
                                    torch.softmax(teacher_logits / T, dim=-1).detach())
            hard_losses = hard_loss(student_logits, labels)
            loss = alpha * soft_losses * T ** 2 + (1 - alpha) * hard_losses

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_epoch += abs(loss.item())
            total_iters += 1

            avg_loss = total_loss_epoch / (i + 1)
            if (i + 1) % 30 == 0 or i + 1 == len(train_loader):
                print(f"\nEpoch {epoch+1}, Iter {i+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
            loss_list.append(avg_loss)
            iter_list.append(total_iters)

            with open(conf.log_path, 'a', encoding='utf-8') as f:
                f.write(f"Epoch {epoch+1}, Iter {i+1}/{len(train_loader)}, Loss: {avg_loss:.4f}\n")

            # 每 150 步评估，保存最优学生模型
            if total_iters % 150 == 0:
                f1, report, cm = model2eval(dev_loader, student_model, device)
                print(f"\nEpoch {epoch+1}, Iter {i+1}/{len(train_loader)}, F1 Score: {f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(student_model.state_dict(), conf.student_best_model_path)
                    with open(conf.log_path, 'a', encoding='utf-8') as f:
                        f.write(f"保存最佳模型，F1 Score: {best_f1:.4f}\n")
                with open(conf.log_path, 'a', encoding='utf-8') as f:
                    f.write(f"Epoch {epoch+1}, Iter {i+1}/{len(train_loader)}, F1 Score: {f1:.4f}\n")
                    f.write(f"Classification Report:\n{report}\n")
                    f.write(f"Confusion Matrix:\n{cm}\n")
                student_model.train()

    # 保存最终学生模型
    torch.save(student_model.state_dict(), conf.student_last_model_path)
    print(f'模型保存成功，f1:{best_f1:.4f}')
    with open(conf.log_path, 'a', encoding='utf-8') as f:
        f.write(f'模型保存成功，f1:{best_f1:.4f}\n')

    plt.plot(iter_list, loss_list)
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('./loss.png')


def main():
    set_seed(conf.seed)
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    # 加载老师模型（训练好的 BERT）
    teacher_model = TMFBertClassifier().to(conf.device)
    teacher_model.load_state_dict(torch.load(conf.teacher_best_model_path))
    # 学生模型（BiLSTM）
    student_model = BiLSTM().to(conf.device)
    model2distill(train_dataloader, dev_dataloader, teacher_model, student_model, conf.device)

    # 测试集最终评估
    best_student_model = BiLSTM().to(conf.device)
    best_student_model.load_state_dict(torch.load(conf.student_best_model_path))
    f1, report, confmat = model2eval(test_dataloader, best_student_model, conf.device)
    with open(conf.log_path, 'a', encoding='utf-8') as f:
        f.write(f'test f1:{f1:.4f}\n')
        f.write(f'test report:{report}\n')
        f.write(f'test confmat:{confmat}\n')


if __name__ == '__main__':
    main()
