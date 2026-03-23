class Trainer():
    def __init__(
            self,
            model,
            optimizer,
            loss_fn,
            trainer_loader,
            val_loader,
            device,
            save_dir
        ):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn  
        self.trainer_loader = trainer_loader
        self.val_loader = val_loader
        self.device = device 
        self.model = self.model.to(self.device)
        self.save_dir = save_dir

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for Batch, (X_batch, Y_batch) in enumerate(self.trainer_loader):
            if (Batch % 5000 == 0):
                print(f"Batch: [{Batch}/{len(self.trainer_loader)}]")
            X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

            preds = self.model(X_batch).squeeze()
            loss = self.loss_fn(preds, Y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(self.trainer_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")
        return self.model

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in self.val_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                pred = self.model(X_batch).squeeze()
                loss = self.loss_fn(pred, Y_batch)

                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss

    def train_epochs(self, epochs):
        best_val_loss = torch.inf
        for i in range(epochs):
            model = self.train_one_epoch(i+1)
            val_loss_epoch = self.evaluate()
            if val_loss_epoch < best_val_loss: # Checkpointing best val loss
                best_val_loss = val_loss_epoch
                torch.save(model.state_dict(), os.path.join(self.save_dir, 'rnn.pth'))
            print("-"*30)