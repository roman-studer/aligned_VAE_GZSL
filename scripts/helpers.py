import torch


def reparameterize(mu, logvar, config):
    """
    Re-parameterization trick to sample from N(mu, var)
    :param mu: vector of means
    :param logvar: vector of log variances
    :param config: config object
    :return: sample from N(mu, var)
    """
    if config.reparameterize_with_noise:
        sigma = torch.exp(logvar)
        eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps
    else:
        return mu


def map_label(label, classes, config):
    """
    Mapping label to int in range [0, classes-1]
    :param label: array of labels
    :param classes: array of classes
    :param config: config object
    :return:
    """
    mapped_label = torch.LongTensor(label.size()).to(config.device)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label
