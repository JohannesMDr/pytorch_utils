class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit = nn.CrossEntropyLoss()
        
    def forward(self, pred, batch):
        loss = {}
        loss['root'] = self.crit(pred['root'], batch['root'])
        loss['vowel'] = self.crit(pred['vowel'], batch['vowel'])
        loss['consonant'] = self.crit(pred['consonant'], batch['consonant'])
        loss_all = loss['root'] + loss['vowel'] + loss['consonant']
        return loss_all, loss
