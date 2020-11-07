import numpy as np
import libmr

def weibull_tailfitting(mean, distance, labellist, tailsize=20, distance_type='eucos'):
    weibull_model = {}
    for category in labellist:
        weibull_model[category] = {}
        distance_scores = np.array(distance[category][distance_type])
        meantrain_vec = np.array(mean[category])
        weibull_model[category]['distances_%s' % distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        weibull_model[category]['weibull_model'] = []
        mr = libmr.MR()
        tailtofit = sorted(distance_scores)[-tailsize:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] += [mr]
    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec']]
    category_weibull += [weibull_model[category_name]['distances_%s' % distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull
