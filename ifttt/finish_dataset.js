
const fs = require('fs');
const byline = require('byline');

function makeArg(name, value, type, op) {
    return { name: { id: 'tt:param.' + name }, value: { value: value }, type: type,
             operator: op };
}

function findEntity(entities, tag) {
    var current = null;
    for (var candidate of entities) {
        var match = /^([A-Z\_]+)="([^"]*)"$/.exec(candidate);
        if (match === null) {
            console.log('Badly formatted entity tag: ' + candidate);
            continue;
        }
        if (match[1] === tag) {
            if (current !== null) {
                console.log('Duplicate required entity ' + tag);
                return null;
            }
            current = match[2];
        }
    }
    return current;
}

var cnt = 0;
function handleOne(map, line, output) {
    var [id, description, ifttt, entities] = line.split(/\t+/);
    
    if (!(ifttt in map)) {
        console.log('Skipped: ' + ifttt + ' not supported');
        return;
    }
    var json = JSON.parse(map[ifttt]);
    if (entities)
        entities = entities.split(',');
    else
        entities = [];
    
    // the map will claim this parses to '@builtin.at() => @$notify', ie
    // "monitor if every day at ...", which is not very useful
    // we want "every day at ... say" instead
    if (ifttt === 'date___time.every_day_at+if_notifications.send_a_notification') {
        json = { rule: { trigger: { name: { id: "tt:builtin.at" }, args: [] },
                         action: { name: { id: "tt:builtin.notify" }, args: [] } } };
    }
    
    if (ifttt.startsWith('date___time.every_day_at+')) {
        var time = findEntity(entities, "TIME");
        if (time && /T[0-9]{2}:[0-9]{2}/.test(time))
            json.rule.trigger.args.push(makeArg("time", time.substr(1), "Time", "is"));
    }
    if (ifttt.endsWith('+gmail.send_an_email')) {
        var email = findEntity(entities, "EMAIL_ADDRESS");
        if (email)
            json.rule.action.args.push(makeArg("to", email, "EmailAddress", "is"));
    }
    if (ifttt.startsWith('instagram.new_photo_by_you_with_specific_hashtag+') ||
        ifttt.startsWith('twitter.new_tweet_by_you_with_hashtag+')) {
       var hashtag = findEntity(entities, "HASHTAG");
       if (!hashtag) {
           console.log('Skipped: cannot find required hashtag');
           return;
       }
       if (json.rule)
           json.rule.trigger.args.push(makeArg("hashtags", hashtag, "String", "has"));
       else
           json.trigger.args.push(makeArg("hashtags", hashtag, "String", "has"));
    }
    if (ifttt.startsWith('twitter.new_tweet_by_a_specific_user+')) {
       var user = findEntity(entities, "USER");
       if (!user) {
           console.log('Skipped: cannot find required username');
           return;
       }
       if (json.rule)
           json.rule.trigger.args.push(makeArg("from", user, "String", "is"));
       else
           json.trigger.args.push(makeArg("from", user, "String", "is"));
    }
    
    // I self identify as an IFTTT rule, my pronouns are...
    description = description.replace(/\byourself\b/g, 'myself');
    description = description.replace(/\byour\b/g, 'my');
    description = description.replace(/\byou\b/g, 'me');
    
    // this is a little aggressive and will filter out emojis
    // too bad
    const NON_ASCII = /[^\x00-\x7f]/;
    if (NON_ASCII.test(description)) {
        console.log('Skipped: description does not look like English');
        return;
    }
    
    //console.log('Succesfully transformed ' + id + ' (' + (++cnt) + ')');
    output.write(description + '\t' + JSON.stringify(json) + '\n');
}

function main() {
    // load the ifttt-to-json.tsv map
    var mapFile = fs.readFileSync(process.argv[2]).toString('utf8');
    var map = {};
    for (var line of mapFile.split(/\n/)) {
        if (!line)
            continue;
        var split = line.split(/\t+/);
        map[split[0]] = split[1];
    }
    
    // now load the actual data
    var input = byline(fs.createReadStream(process.argv[3]));
    input.setEncoding('utf8');
    var output = fs.createWriteStream(process.argv[4]);
    output.setDefaultEncoding('utf8');

    input.on('data', function(line) {
        handleOne(map, line, output);
    });
    input.on('end', function() { output.end(); });
}
main();
