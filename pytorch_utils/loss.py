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


    
def ohem_loss( rate, cls_pred, cls_target ):
    batch_size = cls_pred.size(0) 
    ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate) )
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss
