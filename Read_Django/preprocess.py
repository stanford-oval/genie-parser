import re
import nltk


def preprocess_dataset(annot_file, code_file):
    f_annot = open('./raw_data/annot.all.canonicalized.txt', 'w')
    f_code = open('./raw_data/code.all.canonicalized.txt', 'w')

    examples = []

    err_num = 0
    for idx, (annot, code) in enumerate(zip(open(annot_file), open(code_file))):
        annot = annot.strip()
        code = code.strip()
        try:
            clean_query_tokens, clean_code, str_map = canonicalize_example(annot, code)
            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code,
                       'str_map': str_map, 'raw_code': code}
            examples.append(example)

            f_annot.write('example# %d\n' % idx)
            f_annot.write(' '.join(clean_query_tokens) + '\n')
            f_annot.write('%d\n' % len(str_map))
            for k, v in str_map.iteritems():
                f_annot.write('%s ||| %s\n' % (k, v))

            f_code.write('example# %d\n' % idx)
            f_code.write(clean_code + '\n')
        except:
            print code
            err_num += 1

        idx += 1

    f_annot.close()
    f_annot.close()

    # serialize_to_file(examples, 'django.cleaned.bin')

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples



def canonicalize_example(query, code):
    from lang.py.parse import parse_raw, parse_tree_to_python_ast, canonicalize_code as make_it_compilable
    import astor, ast

    canonical_query, str_map = canonicalize_query(query)
    canonical_code = code

    for str_literal, str_repr in str_map.iteritems():
        canonical_code = canonical_code.replace(str_literal, '\'' + str_repr + '\'')

    canonical_code = make_it_compilable(canonical_code)

    # sanity check
    parse_tree = parse_raw(canonical_code)
    gold_ast_tree = ast.parse(canonical_code).body[0]
    gold_source = astor.to_source(gold_ast_tree)
    ast_tree = parse_tree_to_python_ast(parse_tree)
    source = astor.to_source(ast_tree)

    assert gold_source == source, 'sanity check fails: gold=[%s], actual=[%s]' % (gold_source, source)

    query_tokens = canonical_query.split(' ')

    return query_tokens, canonical_code, str_map



"""
Parses single or double-quoted strings while preserving escaped quote chars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. captures the first quote character (either a single-quote or double-quote)
   and stores it in a group named "quote"
2. takes zero or more characters up until it matches the character that is
   stored in the group named "quote" (i.e. named referenced group) providing
   that the matching "quote" character is not escaped (preceeded by a backslash
   character, i.e. negative look behind assertion).  All matching characters
   between the quotes are stored in a group named "string".
"""

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

def canonicalize_query(query):
    """
    canonicalize the query, replace strings to a special place holder
    """
    str_count = 0
    str_map = dict()

    matches = QUOTED_STRING_RE.findall(query)
    # de-duplicate
    cur_replaced_strs = set()
    for match in matches:
        # If one or more groups are present in the pattern,
        # it returns a list of groups
        quote = match[0]
        str_literal = quote + match[1] + quote

        if str_literal in cur_replaced_strs:
            continue

        # FIXME: substitute the ' % s ' with
        if str_literal in ['\'%s\'', '\"%s\"']:
            continue

        str_repr = '_STR:%d_' % str_count
        str_map[str_literal] = str_repr

        query = query.replace(str_literal, str_repr)

        str_count += 1
        cur_replaced_strs.add(str_literal)

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls like foo.bar.func
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    query = ' '.join(new_query_tokens)

    return query, str_map


if __name__ == '__main__':

    annot_file = '/Users/Mehrad/Documents/GitHub/Read_Django/raw_data/all.anno'
    code_file = '/Users/Mehrad/Documents/GitHub/Read_Django/raw_data/all.code'


    preprocess_dataset(annot_file, code_file)