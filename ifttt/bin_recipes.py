#!/usr/bin/python3

import os
import sys
import re
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('GTK3Cairo')
import matplotlib.pyplot as plt

SUPPORTED_CHANNELS=set(['if_notifications', 'date___time', 'email',
    'twitter', 'android_device', 'dropbox', 'sms', 'philips_hue',
    'facebook', 'gmail', 'tumblr', 'instagram', 'weather', 'android_sms',
    'space', 'stocks', 'up_by_jawbone', 'slack', 'youtube', 'linkedin'])
FILTER_BY_CHANNEL=True

def normalize(x):
    return re.sub('[^a-z0-9]', '_', x.strip().lower().decode('utf-8'))

def sort_counter(counter):
    data = [None] * len(counter)
    i = 0
    for k, v in counter.items():
        data[i] = (k, v)
        i += 1
    
    data.sort(key=lambda x: -x[1])
    labels = list(map(lambda x: x[0], data))
    counts = list(map(lambda x: x[1], data))
    return labels, counts

def plot_counter(data):
    labels, counts = data
    
    ind = np.arange(len(labels))
    width = 0.35
    plt.bar(ind, counts, 0.35)
    plt.xticks(ind + width*.5, labels)

def plot_cdf(axis, data):
    labels, counts = data
    
    sum = np.sum(counts)
    normalized = np.cumsum(counts)/sum
    
    ind = np.arange(len(labels))
    axis.plot(ind, normalized)
    
    level = np.full((len(labels),), 0.9)
    axis.plot(ind, level, '--r')
    
    distance = np.abs(normalized - 0.9)
    quantile = np.argmin(distance)
    
    axis.axvline(quantile, color='r', linestyle='--')
    
    plt.annotate(str(quantile), (quantile+1, 0.89), (quantile + 10, 0.7), arrowprops=dict(facecolor='black', shrink=0.05))
    
    #ticks = axis.get_xticks()
    #labels = axis.get_xticklabels()
    
    #for i in range(len(ticks)):
    #    ticks[i] = (ticks[i], labels[i])
    #ticks.append((quantile, str(quantile)))
    #sort(ticks, key=lambda x: x[0])
    
    #ticklocs = list(map(lambda x: x[0], ticks))
    #labels = list(map(lambda x: x[1], ticks))
    #axis.set_xticks(ticklocs)
    #axis.set_xticklabels(labels)
    
    #plt.xticks(ind + .5, labels)

def print_top(data, how_many, title):
    print('### ' + (title % (how_many,)))
    
    sum = 0
    for i in range(min(how_many, len(data[0]))):
        sum += data[1][i]
        print('%d) %s: %d' % (i+1, data[0][i], data[1][i]))
    print("Total: %d" % (sum,))

def filter_channels(data, how_many):
    labels, counts = data
    new_labels, new_counts = [], []
    
    for i in range(min(how_many, len(labels))):
        full_ifaces = labels[i].split('+')
        ok = True
        for full_iface in full_ifaces:
            channel, iface = full_iface.split('.')
            if channel not in SUPPORTED_CHANNELS:
                ok = False
                break
        if ok:
            new_labels.append(labels[i])
            new_counts.append(counts[i])
    
    return new_labels, new_counts

def convert_shares(x):
    if x.endswith(b'k'):
        return int(1000 * float(x[:-1]))
    else:
        return int(x)

def convert_id(x):
    return int(x.strip().replace(b'\xef\xbb\xbf', b''))

def main():
    trigger_recipes = Counter()
    action_recipes = Counter()
    channel_recipes = Counter()
    joint_recipes = Counter()
    trigger_shares = Counter()
    action_shares = Counter()
    channel_shares = Counter()

    data = np.genfromtxt(sys.argv[1], delimiter='\t', comments='\0', converters=dict(id=convert_id, shares=convert_shares),
        dtype=None, names=True)
    print(data)
    for line in data:
        trigger_channel = normalize(line['triggerchannel'])
        action_channel = normalize(line['actionchannel'])
        trigger = normalize(line['trigger'])
        action = normalize(line['action'])
        shares = line['shares']
            
        trigger_recipes[trigger_channel + '.' + trigger] += 1
        trigger_shares[trigger_channel + '.' + trigger] += shares
        action_recipes[action_channel + '.' + action] += 1
        action_shares[action_channel + '.' + action] += shares
            
        channel_recipes[trigger_channel] += 1
        channel_shares[trigger_channel] += shares
        channel_recipes[action_channel] += 1
        channel_shares[action_channel] += shares
        
        joint_recipes[trigger_channel + '.' + trigger + '+' + action_channel + '.' + action] += 1
    
    #with open('ifttt-stats.json', 'w') as f:
    #    json.dump({ 'trigger_recipes': trigger_recipes,
    #                'action_recipes': action_recipes,
    #                'channel_recipes': channel_recipes,
    #                'trigger_shares': trigger_shares,
    #                'action_shares': action_shares,
    #               'channel_shares': channel_shares }, f)

    channel_recipes = sort_counter(channel_recipes)
    trigger_recipes = sort_counter(trigger_recipes)
    action_recipes = sort_counter(action_recipes)
    joint_recipes = sort_counter(joint_recipes)
    
    if FILTER_BY_CHANNEL:
        trigger_recipes = filter_channels(trigger_recipes, 132)
        action_recipes = filter_channels(action_recipes, 54)
        joint_recipes = filter_channels(joint_recipes, 3468)
    
    print_top(channel_recipes, 22, "Top %d channels by # of recipes")
    print_top(trigger_recipes, 132, "Top %d triggers by # of recipes")
    print_top(action_recipes, 54, "Top %d actions by # of recipes")
    #print_top(joint_recipes, 200, "Top %d trigger-action pairs by # of recipes")
    
    #plt.xkcd()

    #fig1 = plt.figure()
    
    #plt.subplot(3, 1, 1)
    #plt.title('# recipes by channel')
    #plot_counter(channel_recipes)

    #plt.subplot(3, 1, 2)
    #plt.title('# recipes by trigger')
    #plot_counter(trigger_recipes)

    #plt.subplot(3, 1, 3)
    #plt.title('# recipes by action')
    #plot_counter(action_recipes)
    
    #fig2 = plt.figure()
    #ax = plt.subplot(2, 2, 1)
    #plt.title('Recipe distribution by channel')
    #plot_cdf(ax, channel_recipes)
    
    #ax = plt.subplot(2, 2, 2)
    #plt.title('Recipe distribution by trigger')
    #plot_cdf(ax, trigger_recipes)
    
    #ax = plt.subplot(2, 2, 3)
    #plt.title('Recipe distribution by action')
    #plot_cdf(ax, action_recipes)
    
    #ax = plt.subplot(2, 2, 4)
    #plt.title('Joint distribution by trigger-action')
    #plot_cdf(ax, joint_recipes)

    #plt.show()
    
main()
