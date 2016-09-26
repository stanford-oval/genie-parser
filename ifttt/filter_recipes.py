#!/usr/bin/python3

import os
import sys
import re
import json
from collections import Counter

import numpy as np

SUPPORTED_CHANNELS=set(['if_notifications', 'date___time', 'email',
    'twitter', 'android_device', 'dropbox', 'sms', 'philips_hue',
    'facebook', 'gmail', 'tumblr', 'instagram', 'weather', 'android_sms',
    'space', 'stocks', 'up_by_jawbone', 'slack', 'youtube', 'linkedin'])
SUPPORTED_TRIGGERS=set(['instagram.any_new_photo_by_you', 'date___time.every_day_at', 'twitter.new_tweet_by_you', 'gmail.any_new_email_in_inbox',
    'android_sms.any_new_sms_received', 'space.astronomy_picture_of_the_day_by_nasa', 'up_by_jawbone.new_sleep_logged'
    'instagram.new_photo_by_you_with_specific_hashtag', 'twitter.new_tweet_by_a_specific_user',
    'twitter.new_tweet_by_you_with_hashtag'])
SUPPORTED_ACTIONS=set(['twitter.post_a_tweet', 'email.send_me_an_email', 'sms.send_me_an_sms', 'if_notifications.send_a_notification',
    'twitter.post_a_tweet_with_image', 'gmail.send_an_email', 'facebook.create_a_status_message', 'tumblr.create_a_photo_post',
    'facebook.upload_a_photo_from_url', 'tumblr.create_a_text_post', 'android_device.set_ringtone_volume', 'android_sms.send_an_sms'
    'slack.post_to_channel', 'android_device.mute_ringtone', 'linkedin.share_a_link'])

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

def print_top(data, how_many, title):
    how_many = min(how_many, len(data[0]))
    print('### ' + (title % (how_many,)))
    
    sum = 0
    for i in range(how_many):
        sum += data[1][i]
        print('%d) %s: %d' % (i+1, data[0][i], data[1][i]))
    print("Total: %d" % (sum,))

def filter_supported(data, how_many):
    labels, counts = data
    new_labels, new_counts = [], []
    
    for i in range(min(how_many, len(labels))):
        trigger, action = labels[i].split('+')
        if trigger not in SUPPORTED_TRIGGERS:
            #print('Skipped trigger ' + trigger)
            continue
        if action not in SUPPORTED_ACTIONS:
            #print('Skipped action ' + action)
            continue
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
    joint_recipes = filter_supported(joint_recipes, len(joint_recipes[0]))
    
    print_top(joint_recipes, 200, "Top %d trigger-action pairs by # of recipes")
    
    with open(sys.argv[2], 'w', encoding='utf-8') as out:
        for line in data:
            trigger_channel = normalize(line['triggerchannel'])
            action_channel = normalize(line['actionchannel'])
            trigger = trigger_channel + '.' + normalize(line['trigger'])
            action = action_channel + '.' + normalize(line['action'])

            if trigger not in SUPPORTED_TRIGGERS or action not in SUPPORTED_ACTIONS:    
                continue
            print(str(line['id']) + '\t' + line['description'].decode('utf-8') + '\t' + trigger + '+' + action, file=out)
    
main()
