const fs = require('fs');
const byline = require('byline');
const mysql = require('mysql');

const URL = 'mysql://ifttt:ifttt@localhost/ifttt';

function makeCanonical(channel, iface) {
    return (iface + ' on ' + channel).toLowerCase();
}
function normalize(str) {
    return str.toLowerCase().replace(/[^a-z0-9]/g, '_');
}

function main() {
    //var conn = mysql.createConnection(URL);
    //conn.connect();
    //conn.query("truncate interfaces");
    //conn.query("start transaction");
    
    var stream = byline(fs.createReadStream(process.argv[2]));
    stream.setEncoding('utf8');
    
    var output = fs.createWriteStream('./ifttt/canonicals.tsv');
    output.setDefaultEncoding('utf8');
    
    var n = 0;
    stream.on('data', function(line) {
        // ignore header
        if (n++ === 0)
            return;
        var [id, description, author, date, shares, triggerChannel, trigger,
             triggerDescription, actionChannel, action, actionDescription] = line.split('\t');
        if (!trigger || !triggerChannel || !action || !actionChannel) {
            console.log('Ignored recipe ' + id);
            return;
        }
        
        var trigCanonical = makeCanonical(triggerChannel, trigger);
        //conn.query("insert ignore into interfaces(channel, name, type, canonical) values (?, ?, 'trigger', ?)",
        //           [normalize(triggerChannel), normalize(trigger), trigCanonical]);
        var actCanonical = makeCanonical(actionChannel, action);
        //conn.query("insert ignore into interfaces(channel, name, type, canonical) values (?, ?, 'action', ?)",
        //           [normalize(actionChannel), normalize(action), actCanonical]);
        
        var rule = { rule: { trigger: { name: { id: "tt:" + normalize(triggerChannel) + '.' + normalize(trigger) }, args: [] },
                             action: { name: { id: "tt:" + normalize(actionChannel) + '.' + normalize(action) }, args: [] } } };
        output.write(description + '\t' + 'if ' + trigCanonical + ' then ' + actCanonical + '\t' + JSON.stringify(rule) + '\n');
    });
    stream.on('end', function() {
        output.end();
        //conn.query('commit');
        //conn.end();
    });
}
main();
