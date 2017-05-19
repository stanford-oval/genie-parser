#!/usr/bin/env python

import sys, csv, json, random

fp = open(sys.argv[1], 'r')
rdr = csv.reader(fp)
examples = []
for row in rdr:
    examples.append((row[6], row[7]))
fp.close()

fp2 = open(sys.argv[2], 'r')
devices = fp2.read().splitlines()
fp2.close()

fp3 = open(sys.argv[3], 'r')
names = fp3.read().splitlines()
fp3.close()

fp4 = open(sys.argv[4], 'r')
verbs = fp4.read().splitlines()
fp4.close()

tell_words = ['tell', 'ask', 'suggest', 'inform', 'advise', 'recommend', 'request']
setup_words = ['setup', 'send'] + tell_words

mixing_prob = 0.3

for (utterance, target_json) in examples:
    json_obj = json.loads(target_json)
    name, gender = random.choice(names).split(',')
    gender = gender.strip()

    my_pronoun = "her" if gender is 'F' else "his"
    i_pronoun = "she" if gender is 'F' else "he"
    me_pronoun = "her" if gender is 'F' else "him"

    name2, gender2 = random.choice(names).split(',')
    pronoun2 = "her" if gender2 is 'F' else "his"

    new_utterance = ""
    if "rule" in json_obj.keys():
        devs = {}
        index = []
        for key in json_obj["rule"].keys():
            devs[key] = json_obj["rule"][key]["name"]["id"].split(':')[1].split('.')[0]
            if devs[key] in [device.lower() for device in devices]:
                search_key = devs[key]
                if devs[key] == 'security-camera':
                    search_key = 'security'
                try:
                    index.append((key, utterance.lower().split().index(search_key)))
                except ValueError:
                    pass

        if len(index) == 2:
            # person_splits: 0.3, 0.3, 0.4
            person1_mask = False
            person2_mask = False
            if random.random() < 0.4:
                person1_mask = True
                person2_mask = True
            else:
                if random.random() < 0.5:
                    person1_mask = True
                else:
                    person2_mask = True

            sindex = sorted(index, key=lambda tup: tup[1])
            split1 = utterance.split()[0:sindex[0][1]+1]
            split2 = utterance.split()[sindex[0][1]+1:sindex[1][1]+1]
            split3 = utterance.split()[sindex[1][1]+1:len(utterance)]

            new_split1 = split1
            double_case = False
            modified = False
            if person1_mask:
                for it, word in enumerate(split1):
                    if word.lower() == 'i':
                        if double_case:
                            new_split1[it] = i_pronoun
                        else:
                            new_split1[it] = "@" + name
                            double_case = True
                            modified = True

                    if word.lower() == 'me':
                        if double_case:
                            new_split1[it] = me_pronoun
                        else:
                            new_split1[it] = "@" + name
                            double_case = True
                            modified = True

                    if word.lower() == 'you':
                        if double_case:
                            new_split1[it] = i_pronoun
                        else:
                            new_split1[it] = "@" + name
                            double_case = True
                            modified = True

                    if word.lower() == 'my':
                        if double_case:
                            new_split1[it] = my_pronoun
                        else:
                            new_split1[it] = "@" + name + "'s"
                            double_case = True
                            modified = True

                    if not double_case and word.lower() in devs.values():
                        if random.random() < mixing_prob:
                            new_split1[it] = word + " of @" + name
                        else:
                            new_split1[it] = "@" + name + "'s " + word
                        modified = True

                if modified == True:
                    key = sindex[0][0]
                    json_obj["rule"][key]["person"] = name.lower()

            new_split2 = split2
            double_case = False
            modified2 = False
            if person2_mask:
                for it, word in enumerate(split2):
                    if word.lower() == 'i':
                        new_split2[it] = "@" + name2
                        double_case = True
                        modified2 = True

                    if word.lower() == 'me':
                        if double_case:
                            new_split2[it] = me_pronoun
                        else:
                            new_split2[it] = "@" + name
                            double_case = True
                            modified = True

                    if word.lower() == 'you':
                        if double_case:
                            new_split2[it] = i_pronoun
                        else:
                            new_split2[it] = "@" + name
                            double_case = True
                            modified = True

                    if word.lower() == 'my':
                        if double_case:
                            new_split2[it] = my_pronoun
                        else:
                            new_split2[it] = "@" + name2 + "'s"
                            double_case = True
                            modified2 = True

                    if not double_case and word.lower() in devs.values():
                        if random.random() < mixing_prob:
                            new_split2[it] = word + " of @" + name2
                        else:
                            new_split2[it] = "@" + name2 + "'s " + word
                        modified2 = True

                if modified2 == True:
                    key = sindex[1][0]
                    json_obj["rule"][key]["person"] = name2.lower()

            if modified or modified2:
                new_utterance = "%s %s %s" % (' '.join(new_split1), ' '.join(new_split2), ' '.join(split3))
                print new_utterance, "\t", json.dumps(json_obj)

        elif len(index) == 1:
            key = index[0][0]
            split1 = utterance.split()[0:index[0][1]+1]
            split2 = utterance.split()[index[0][1]+1:len(utterance)]

            new_split1 = split1
            double_case = False
            modified = True
            for it, word in enumerate(split1):
                if word.lower() == 'i':
                    new_split1[it] = "@" + name
                    double_case = True
                    modified = True

                if word.lower() == 'me':
                    if double_case:
                        new_split1[it] = me_pronoun
                    else:
                        new_split1[it] = "@" + name
                        double_case = True
                        modified = True

                if word.lower() == 'you':
                    if double_case:
                        new_split1[it] = i_pronoun
                    else:
                        new_split1[it] = "@" + name
                        double_case = True
                        modified = True

                if word.lower() == 'my':
                    if double_case:
                        new_split1[it] = my_pronoun
                    else:
                        new_split1[it] = "@" + name + "'s"
                        double_case = True
                        modified = True

                if not double_case and word.lower() == devs[key]:
                    if random.random() < mixing_prob:
                        new_split1[it] = word + " of @" + name
                    else:
                        new_split1[it] = "@" + name + "'s " + word
                    modified = True

            if modified == True:
                json_obj["rule"][key]["person"] = name.lower()
                new_utterance = "%s %s" % (' '.join(new_split1), ' '.join(split2))
                print new_utterance, "\t", json.dumps(json_obj)
        else:
            if utterance.split()[0].lower() in verbs:
                setup_word = random.choice(setup_words)
                if setup_word == 'setup' or setup_word == 'send':
                    new_utterance = "%s @%s a rule: %s" % (setup_word, name, utterance.lower())
                else:
                    new_utterance = "%s @%s to %s" % (setup_word, name, utterance.lower())

                new_utterance = new_utterance.replace(" i ", " " + i_pronoun + " ")
                new_utterance = new_utterance.replace(" me ", " " + me_pronoun + " ")
                new_utterance = new_utterance.replace(" my ", " " + my_pronoun + " ")
                new_utterance = new_utterance.replace(" you ", " " + i_pronoun + " ")

                new_json_obj = {}
                new_json_obj["setup"] = {}
                new_json_obj["setup"]["person"] = name.lower()
                new_json_obj["setup"]["rule"] = json_obj
                
                print new_utterance, "\t", json.dumps(new_json_obj)
            else:
                print utterance, "\t", json.dumps(json_obj)

    else: # Primitive
        double_case = False
        utterance = utterance.lower()
        if " i " in utterance.lower():
            double_case = True

        if " my " in utterance.lower():
            if double_case:
                new_utterance = utterance.replace(" my ", " " + my_pronoun + " ")
            else:
                new_utterance = utterance.replace(" my ", " @" + name + "'s ")
                double_case = True
        else:
            in_check = [" in " + device.lower() in utterance.lower() for device in devices]
            if any(in_check):
                if double_case:
                    new_utterance = utterance.replace(" in ", " in " + my_pronoun + " ")
                else:
                    if random.random() < mixing_prob:
                        device = devices[in_check.index(True)].lower()
                        try:
                            ind = utterance.lower().split().index(device)
                            device_utt = utterance.split()[ind]
                            new_utterance = utterance.replace(" in " + device_utt , " in " + device_utt + " of @" + name )
                        except ValueError:
                            pass
                    else:
                        new_utterance = utterance.replace(" in ", " in @" + name + "'s ")
            on_check = [" on " + device.lower() in utterance.lower() for device in devices]
            if any(on_check):
                if double_case:
                    new_utterance = utterance.replace(" on ", " on " + my_pronoun + " ")
                else:
                    if random.random() < mixing_prob:
                        device = devices[on_check.index(True)].lower()
                        try:
                            ind = utterance.lower().split().index(device)
                            device_utt = utterance.split()[ind]
                            new_utterance = utterance.replace(" on " + device_utt , " on " + device_utt + " of @" + name )
                        except ValueError:
                            pass
                    else:
                        new_utterance = utterance.replace(" on ", " on @" + name + "'s ")
        new_utterance = new_utterance.replace(" I ", " @" + name + " ").replace(" i ", " @" + name + " ")

        if new_utterance != "":
            for prim in json_obj.keys():
                json_obj[prim]['person'] = name.lower()

            print new_utterance, "\t", json.dumps(json_obj)
        else:
            if utterance != "" and utterance.split()[0].lower() in verbs:
                tell_word = random.choice(tell_words)
                new_utterance = "%s @%s to %s" % (tell_word, name, utterance)

                new_utterance = new_utterance.replace(" i ", " " + i_pronoun + " ")
                new_utterance = new_utterance.replace(" me ", " " + me_pronoun + " ")
                new_utterance = new_utterance.replace(" my ", " " + my_pronoun + " ")

                new_json_obj = {}
                new_json_obj["setup"] = {}
                new_json_obj["setup"]["person"] = name.lower()
                for prim in json_obj.keys():
                    new_json_obj["setup"][prim] = json_obj[prim]

                #print >>sys.stderr, utterance
                #print >>sys.stderr, new_utterance
                #print >>sys.stderr, json.dumps(new_json_obj)
                #print >>sys.stderr, ""

                print new_utterance, "\t", json.dumps(new_json_obj)
            else:
                print utterance, "\t", json.dumps(json_obj)
