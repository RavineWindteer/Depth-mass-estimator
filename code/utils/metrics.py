import torch

def compute_metrics_mass(pred, target):

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    alde = torch.mean(torch.abs(diff_log))
    ape = torch.mean(torch.abs(diff / target))
    mnre = torch.min(torch.mean(pred / target), torch.mean(target / pred))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))
    q = float(torch.max(torch.mean(pred / target), torch.mean(target / pred)) < 2.0)

    return {'alde': 0.0 if torch.isnan(alde).item() else alde.item(),
        'ape': 0.0 if torch.isnan(ape).item() else ape.item(),
        'mnre': 0.0 if torch.isnan(mnre).item() else mnre.item(),
        'rmse_log': 0.0 if torch.isnan(rmse_log).item() else rmse_log.item(), 
        'log10': 0.0 if torch.isnan(log10).item() else log10.item(), 
        'silog': 0.0 if torch.isnan(silog).item() else silog.item(),
        'q': q}

def compute_metrics_pc(pred, target):
    return None
