'''
Created on Apr 24, 2018

@author: gcampagn
'''

from .shift_reduce_grammar import ShiftReduceGrammar

from collections import OrderedDict
from keyword import kwlist


class DjangoGrammar(ShiftReduceGrammar):
    '''
    The grammar of Django
    '''

    def __init__(self, filename=None, **kw):
        super().__init__(**kw)
        self.entities = []

        '''
        Python2.7 Grammar
        See this link: https://docs.python.org/2/reference/grammar.html
        '''

        GRAMMAR = OrderedDict({
            '$single_input': [('$simple_stmt',), ('$compound_stmt',)],
            '$file_input': [('$stmt',), ('$file_input', '$stmt')],
            '$eval_input': [('$testlist',)],

            '$decorator': [('@', '$dotted_name'), ('@', '$dotted_name', '(', '$arglist', ')')],
            '$decorators': [('$decorator',), ('$decorators', '$decorator')],
            '$decorated': [('$decorators', '$classdef'), ('$decorators', '$funcdef')],
            '$funcdef': [('def', '$ident', '$parameters', ':', '$suite')],
            '$parameters': [('(', '$varargslist', ')'), ('(', ')')],

            '$var1': [('$fpdef', '=', '$test', ','), ('$var1', '$fpdef', '=', '$test', ',')],
            '$var2': [('**', '$ident'), ('*', '$ident'), ('*', '$ident', ',', '**', '$ident')],
            # '$var3': [()],
            '$varargslist': [('$var1',), ('$var2',)],


            '$fpdef': [('$ident',), ('(', '$fplist', ')')],
            '$fplist': [('$fpdef', ','), ('$fpdef',), ('$fpdef', ',', '$fpdef'), ('$fpdef', ',', '$fpdef', ',')],

            '$stmt': [('$simple_stmt',), ('$compound_stmt',)],
            '$simple_stmt_tmp': [('$small_stmt',), ('$simple_stmt_tmp', ';', '$small_stmt')],
            '$simple_stmt': [('$simple_stmt_tmp',), ('$simple_stmt_tmp', ';')],
            '$small_stmt': [('$expr_stmt',), ('$print_stmt',), ('$del_stmt',), ('$pass_stmt',), ('$flow_stmt',), ('$import_stmt',), ('$global_stmt',),
                            ('$exec_stmt',), ('$assert_stmt',)],
            '$expr_stmt': [('$testlist', '$augassign', '$yield_expr'), ('$testlist', '$augassign', '$testlist'),
                           ('$testlist', '='), ('$testlist', '=', '$yield_expr'), ('$testlist', '=', '$testlist')],

            '$augassign': [('+=',), ('-=',), ('*=',), ('/=',), ('%=',), ('&=',), ('|=',), ('^=',), ('<<=',), ('>>=',), ('**=',), ('//=',)],

            '$print_stmt': [('print', ), ('print', '$test'), ('print', '$test', ',', '$test')], #fixme
            '$del_stmt': [('del', '$exprlist')],
            '$pass_stmt': [('pass',)],
            '$flow_stmt': [('$break_stmt',), ('$continue_stmt',), ('$return_stmt',), ('$raise_stmt',), ('$yield_stmt',)],
            '$break_stmt': [('break',)],
            '$continue_stmt': [('continue',)],
            '$return_stmt': [('return', ), ('return', '$testlist')],
            '$yield_stmt': [('$yield_expr',)],
            '$raise_stmt': [('raise',), ('raise', '$test'), ('raise', '$test', ',', '$test'),
                            ('raise', '$test', ',', '$test', ',', '$test')],
            '$import_stmt': [('$import_name',), ('$import_from',)],
            '$import_name': [('import', '$dotted_as_names')],
            ########################
            '$import_from': [],
            '$import_as_name': [('$ident',), ('$ident', 'as', '$ident')],
            '$dotted_as_name': [],
            '$import_as_names': [],
            '$dotted_as_names': [],
            '$dotted_name': [('$ident',), ('$dotted_name', '.', '$ident')],
            '$global_stmt': [('global', '$ident'), ('$global_stmt', ',', '$ident')],
            '$exec_stmt': [('exec', '$expr'), ('exec', '$expr', 'in', '$test')],
            '$assert_stmt': [('assert', '$test'), ('assert', '$test', ',', '$test')],

            '$compound_stmt': [('$if_stmt',), ('$while_stmt',), ('$for_stmt',), ('$try_stmt',), ('$with_stmt',),
                               ('$funcdef',), ('$classdef',), ('$decorated',)],
            '$if_stmt_tmp': [('if', '$test', ':', '$suite'), ('$if_stmt_tmp', 'elif', '$test', ':', "$suite")],
            '$if_stmt': [('$if_stmt_tmp',), ('$if_stmt_tmp', 'else', ':', '$suite')],
            '$while_stmt': [('while', '$test', ':', '$suite'), ('while', '$test', ':', '$suite', 'else', '$suite')],
            '$for_stmt': [('for', '$exprlist', 'in', '$testlist', ':', '$suite'),
                         ('for', '$exprlist', 'in', '$testlist', ':', '$suite', 'else', ':', '$suite')],
            '$try_stmt': [],
            '$with_stmt': [('with', '$with_item', ':', '$suite')],
            '$with_item': [('$test',), ('$test', 'as', '$expr')],
            '$except_clause': [('except',)],
            '$suite': [('$simple_stmt',), ('$stmt',), ('$suite', '$stmt')],
            '$testlist_safe': [('$old_test',)],
            '$old_test': [('$or_test', '$old_lambdef')],
            '$old_lambdef': [('lambda', ':', '$old_test'), ('lambda', '$varargslist', ':', '$old_test')],
            '$test': [('$lambdef',), ('$or_test',), ('$or_test', 'if', '$or_test', 'else', '$test')],
            '$or_test': [('$and_test',), ('$or_test', 'or', '$and_test')], #fixme
            '$and_test': [('$not_test',), ('$and_test', 'and', '$not_test')],
            '$not_test': [('not', '$not_test'), ('not', '$comparison')],
            '$comparison': [('$expr',), ('$comparison', '$comp_op', '$expr')],
            '$comp_op': [('<',), ('>',), ('==',), ('>=',), ('<=',), ('<>',), ('!=',), ('in',), ('not', 'in'), ('is',), ('is', 'not')],
            '$expr': [('$xor_expr',), ('$expr', '|', '$xor_expr')],
            '$xor_expr': [('$and_expr',), ('xor_expr', '^', '$and_expr')],
            '$and_expr': [('$shift_expr',), ('and_expr', '&', '$shift_expr')],
            '$shift_expr': [('$arith_expr',), ('$shift_expr', '<<', '$arith_expr'), ('$shift_expr', '>>', '$arith_expr')], #fixme
            '$arith_expr': [('$term',), ('$arith_expr', '+', '$term'), ('$arith_expr', '-', '$term')],
            '$term': [('$factor',), ('$term', '*', '$factor'), ('$term', '/', '$factor'),
                     ('$term', '%', '$factor'), ('$term', '//', '$factor')],
            '$factor': [('+', '$factor'), ('-', '$factor'), ('~', '$factor'),
                        ('+', '$power'), ('-', '$power'), ('~', '$power')],
            '$power_tmp': [('$atom',), ('$power_tmp', '$trailer')],
            '$power': [('$power_tmp',), ('$power_tmp', '**', '$factor')],
            '$atom': [('(', '$yield_expr', ')'), ('(', '$testlist_comp', ')'), ('[', '$listmaker', ']'),
                     ('{', '$dictorsetmaker','}'), ('`', '$testlist1', '`'), ('$ident',),
                     ('(', ')'), ('[', ']'), ('{', '}')],
            '$listmaker': [('$test', '$list_for'), ('$test', ',', 'test')], #fixme
            '$testlist_comp': [('$test', '$comp_for'), ('$test', ',', 'test')],
            '$lambdef': [('lambda', ':', '$test'), ('lambda', '$varargslist', ':', '$test')],
            '$trailer': [('.', '$ident'), ('(', '$arglist', ')'), ('[', 'subscriptlist', ']')],
            '$subscriptlist_tmp': [('$subscript',), ('$subscriptlist_tmp', '$subscript')],
            '$subscriptlist': [('$subscriptlist_tmp',), ('$subscriptlist_tmp', ',')],
            '$subscript': [('.', '.', '.'), ('$test',)], #fixme
            '$sliceop': [(':',), (':', '$test')],
            '$exprlist_tmp': [('$expr',), ('$exprlist', ',', '$expr')],
            '$exprlist': [('$exprlist_tmp',), ('$exprlist_tmp', ',')],
            '$testlist_tmp': [('$test',), ('$testlist', ',', '$test')],
            '$testlist': [('$testlist_tmp',), ('$testlist_tmp', ',')],
            '$dictorsetmaker': [],

            '$classdef': [('class', '$ident', ':', '$suite'), ('class', '$ident', '(', '$testlist', ')', ':', '$suite')],
            '$arglist_tmp': [('$argument', ','), ('$arglist_tmp', '$argument', ',')],
            '$arglist': [('$arglist_tmp', '$argument'), ('$arglist_tmp', '$argument', ','),
                        ('$arglist_tmp', '*', '$test'), ('$arglist_tmp', '**', '$test')],
            '$argument': [('$test',), ('$test', '$comp_for'), ('$test', '=', '$test')],
            '$list_iter': [('$list_for',), ('$list_if',)],
            '$list_for': [('for', '$exprlist', 'in', '$testlist_safe'),
                          ('for', '$exprlist', 'in', '$testlist_safe', '$list_iter')],
            '$list_if': [('if', '$old_test'), ('if', '$old_test', '$list_iter')],

            '$comp_iter': [('$comp_for',), ('$comp_if',)],
            '$comp_for': [('for', '$exprlist', 'in', '$or_test'), ('for', '$exprlist', 'in', '$or_test', '$comp_iter')],
            '$comp_if': [('if', '$old_test'), ('if', '$old_test', '$comp_iter')],

            '$testlist1': [('$test',), ('$testlist1', ',', '$test')],
            '$yield_expr': [('yield',), ('yield', '$testlist')], #fixme

            '$ident': []
        })
            

        self.allfunctions = set()


        # add common keywords
        kwlist.extend(['name', '__init__', 'from', 'import', 'return', 'def', 'self', 'pass', 'None', 'NotImplementedError', 'io',
                       'os', 'random', 'class', 'utils', 'key', 'value', 'open', 'close', 'pickle', 'isinstance',
                       'LibraryValueNotFoundException'
                       ])

        with open(filename, 'r') as fp:
            #i = 0
            for line in fp:
                #if i >= 2000:
                    #break
                for token in line.strip().split(' '):
                    if token:
                        if token[0].isalpha() and not token in kwlist:
                            if token.find('$') == -1:
                                #i += 1
                                GRAMMAR['$ident'].append((token,))

        self.num_functions = 0
        self.tokens += self.construct_parser(GRAMMAR)

        self.dictionary = dict()
        for i, token in enumerate(self.tokens):
            self.dictionary[token] = i
