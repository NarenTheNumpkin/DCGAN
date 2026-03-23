import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os

class Trainer():
    def __init__(
            self,
            generator,
            discriminator,
            gen_optimizer,
            dis_optimizer,
            trainer_loader,
            device,
            save_dir
        ):
        
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.trainer_loader = trainer_loader
        self.device = device 
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir="logs")

    def train_one_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        criterion = nn.BCELoss()
        total_gen_loss = 0
        total_dis_loss = 0

        for i, (real_images, _) in enumerate(self.trainer_loader):
            real_images = real_images.to(self.device)
            BATCH_SIZE = real_images.size(0)
            labels_real = torch.full((BATCH_SIZE, ), 1.0, device=self.device)
            labels_fake = torch.full((BATCH_SIZE, ), 0.0, device=self.device)

            self.dis_optimizer.zero_grad()

            """
            Training Discriminator
            """

            # train on real
            output_real = self.discriminator(real_images).view(-1)
            loss_d_real = criterion(output_real, labels_real)

            # train on fake
            z = torch.rand(BATCH_SIZE, 100, device=self.device)
            fake = self.generator(z)
            output_fake = self.discriminator(fake.detach()).view(-1) # detach from comp graph otherwise grad will flow back to generator
            loss_d_fake = criterion(output_fake, labels_fake)

            dis_loss += loss_d_real + loss_d_fake
            dis_loss.backward()
            self.dis_optimizer.step()

            """
            Training Generator
            """

            self.gen_optimizer.zero_grad()
            output_g = self.discriminator(fake).view(-1)
            gen_loss = criterion(output_g, labels_real)

            gen_loss.backward()
            self.gen_optimizer.step()

            total_gen_loss += gen_loss.item()
            total_dis_loss += dis_loss.item()
        
            step = epoch * len(self.trainer_loader) + i
            self.writer.add_scalar("Loss/Generator", gen_loss.item(), step)
            self.writer.add_scalar("Loss/Discriminator", dis_loss.item(), step)

        return total_gen_loss/len(self.trainer_loader), total_dis_loss/len(self.trainer_loader)

    def train_epochs(self, epochs):
        for epoch in range(1, epochs + 1):
            avg_gen, avg_dis = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} | Gen Loss: {avg_gen:.4f} | Dis Loss: {avg_dis:.4f}")

            self.generator.eval()
            with torch.no_grad():
                fakes = self.generator(self.fixed_noise).detach().cpu()
                img_grid = torchvision.utils.make_grid(fakes, padding=2, normalize=True)
                
                self.writer.add_image("Generated_Images", img_grid, global_step=epoch)
                img_path = os.path.join(self.save_dir, "plots", f"epoch_{epoch}.png")
                torchvision.utils.save_image(img_grid, img_path)

            torch.save(self.generator.state_dict(), os.path.join(self.save_dir, "model", "generator.pth"))