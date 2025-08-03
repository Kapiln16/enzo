from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item["input"], truncation=True, padding="max_length", return_tensors="pt")
        label = self.tokenizer(item["label"], truncation=True, padding="max_length", return_tensors="pt")
        return encoding.input_ids.squeeze(0), label.input_ids.squeeze(0)

def get_dataloader(dataset, tokenizer, batch_size=8):
    ds = SimpleDataset(dataset, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
