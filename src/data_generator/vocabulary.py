# coding:utf-8
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function


import os
import re
import jieba_fast as jieba
import thulac

from src.utils import utils

jieba.initialize()
thu = thulac.thulac()


PAD, UNK, EOS = copy_head = ['<pad>', '<unk>', '<eos>']


allowed_suffix = [
    'com', 'net', 'org', 'gov', 'mil', 'edu', 'biz', 'info', 'pro', 'name', 'coop',
    'travel', 'xxx', 'idv', 'aero', 'museum', 'mobi', 'asia', 'tel', 'int', 'post',
    'jobs', 'cat', 'ac', 'ad', 'ae', 'af', 'ag', 'ai', 'al', 'am', 'an', 'ao', 'aq',
    'ar', 'as', 'at', 'au', 'aw', 'az', 'ba', 'bb', 'bd', 'be', 'bf', 'bg', 'bh', 'bi',
    'bj', 'bm', 'bn', 'bo', 'br', 'bs', 'bt', 'bv', 'bw', 'by', 'bz', 'ca', 'cc', 'cd',
    'cf', 'cg', 'ch', 'ci', 'ck', 'cl', 'cm', 'cn', 'co', 'cr', 'cu', 'cv', 'cx', 'cy',
    'cz', 'de', 'dj', 'dk', 'dm', 'do', 'dz', 'ec', 'ee', 'eg', 'eh', 'er', 'es', 'et',
    'eu', 'fi', 'fj', 'fk', 'fm', 'fo', 'fr', 'ga', 'gd', 'ge', 'gf', 'gg', 'gh', 'gi',
    'gl', 'gm', 'gn', 'gp', 'gq', 'gr', 'gs', 'gt', 'gu', 'gw', 'gy', 'hk', 'hm', 'hn',
    'hr', 'ht', 'hu', 'id', 'ie', 'il', 'im', 'in', 'io', 'iq', 'ir', 'is', 'it', 'je',
    'jm', 'jo', 'jp', 'ke', 'kg', 'kh', 'ki', 'km', 'kn', 'kp', 'kr', 'kw', 'ky', 'kz',
    'la', 'lb', 'lc', 'li', 'lk', 'lr', 'ls', 'ma', 'mc', 'md', 'me', 'mg', 'mh', 'mk',
    'ml', 'mm', 'mn', 'mo', 'mp', 'mq', 'mr', 'ms', 'mt', 'mu', 'mv', 'mw', 'mx', 'my',
    'mz', 'na', 'nc', 'ne', 'nf', 'ng', 'ni', 'nl', 'no', 'np', 'nr', 'nu', 'nz', 'om',
    'pa', 'pe', 'pf', 'pg', 'ph', 'pk', 'pl', 'pm', 'pn', 'pr', 'ps', 'pt', 'pw', 'py',
    'qa', 're', 'ro', 'ru', 'rw', 'sa', 'sb', 'sc', 'sd', 'se', 'sg', 'sh', 'si', 'sj',
    'sk', 'sm', 'sn', 'so', 'sr', 'st', 'sv', 'sy', 'sz', 'tc', 'td', 'tf', 'tg', 'th',
    'tj', 'tk', 'tl', 'tm', 'tn', 'to', 'tp', 'tr', 'tt', 'tv', 'tw', 'tz', 'ua', 'ug',
    'uk', 'um', 'us', 'uy', 'uz', 'va', 'vc', 've', 'vg', 'vi', 'vn', 'vu', 'wf', 'ws',
    'ye', 'yt', 'yu', 'yr', 'za', 'zm', 'zw']


def strip_line(line):
    return line.strip()


def del_space(line):
    return re.sub(r' +', ' ', line)


def sub_email(line):
    url_re = re.compile(
        r'(([a-zA-Z0-9_-]+\.)*[a-zA-Z0-9_-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z]+)(?![a-zA-Z])')
    new_line = line
    resu = []
    for i in url_re.finditer(line):
        curr = line[i.start(): i.end()]
        for a in allowed_suffix:
            if curr.endswith('.' + a):
                resu.append(line[i.start(): i.end()])
    for r in resu:
        new_line = re.sub(r, r'[邮箱x]', new_line)
    return new_line


def sub_url(line):
    url_re = re.compile(
        r'(([a-zA-Z]+:\\\\|[a-zA-Z]+://)?([a-zA-Z0-9-]+\.)'
        r'+[a-zA-Z]{2,10}[a-zA-Z0-9-_ ./?#%&=]*)(?![a-zA-Z])|'
        r'(([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,10})(?![a-zA-Z])')
    new_line = line
    resu = []
    for i in url_re.finditer(line):
        curr = line[i.start(): i.end()]
        for a in allowed_suffix:
            if curr.endswith('.' + a):
                resu.append(line[i.start(): i.end()])
    for r in resu:
        new_line = re.sub(r, r'[链接x]', new_line)
    return new_line


default_regex = [
    (re.compile(r'\[ORDERID_\d+\]'), 'order_id'),
    (re.compile(r'#E-s\d*\[数字x\]'), 'emoji'),
    (re.compile(r'\[邮箱x\]'), 'email'),
    (re.compile(r'\[数字x\]'), 'number'),
    (re.compile(r'\[地址x\]'), 'location'),
    (re.compile(r'\[时间x\]'), 'time'),
    (re.compile(r'\[日期x\]'), 'date'),
    (re.compile(r'\[链接x\]'), 'link'),
    (re.compile(r'\[电话x\]'), 'phone'),
    (re.compile(r'\[金额x\]'), 'price'),
    (re.compile(r'\[姓名x\]'), 'name'),
    (re.compile(r'\[站点x\]'), 'station'),
    (re.compile(r'\[身份证号x\]'), 'photo_id'),
    (re.compile(r'\[组织机构x\]'), 'organization'),
    (re.compile(r'\[子\]'), 'non-sense'),
    (re.compile(r'\[父原始\]'), 'non-sense'),
    (re.compile(r'\[父\]'), 'non-sense'),
    (re.compile(r'~O\(∩_∩\)O/~'), 'smiler'),
    (re.compile(r'<s>'), 'splitter'),
    (re.compile(r'\d{11}'), 'phone'),
    (re.compile(r'\d{6,10}'), 'number'),
    (re.compile(r'\d{11,15}'), 'number'),
    (re.compile(r'&nbsp;'), 'non-sense')
]


default_clean_fns = [del_space, sub_email, sub_url, strip_line]


class SubCutter(object):
    def __init__(self, regex=default_regex, chinese_seg='jieba', with_tag=True):
        self.collector = None
        self.counter = None
        self.regex = regex
        self.chinese_seg = chinese_seg
        self.with_tag = with_tag

    def _reset(self):
        self.collector = []
        self.counter = 0

    def _line_to_tokens(self, line):
        if self.chinese_seg == 'jieba':
            return jieba.lcut(line)
        elif self.chinese_seg == 'thulac':
            return [i[0] for i in thu.cut(line)]

    def _sub_fn(self, tag, with_tag):

        def sub_with_num(matched):
            if with_tag:
                self.collector.append('{{{{{}:{}}}}}'.format(tag, matched.group()))
            else:
                self.collector.append(matched.group())
            self.counter += 1
            return 'ph_{}_{}_'.format(tag, self.counter)

        return sub_with_num

    def segment(self, line):
        self._reset()
        new_line = line
        for r, ta in self.regex:
            new_line = r.sub(self._sub_fn(ta, self.with_tag), new_line)
        matched = re.finditer(r'ph_[\s\S]+?_(?P<num>\d{1,2})_', new_line)
        start = 0
        tokens = []
        for m in matched:
            tokens += self._line_to_tokens(new_line[start: m.start()])
            tokens += [self.collector[int(m.group('num')) - 1]]
            start = m.end()
        tokens += jieba.lcut(new_line[start:])
        return tokens

    def cut(self, line):
        new_line = line
        for fn in default_clean_fns:
            new_line = fn(new_line)
        return self.segment(new_line)


class Tokenizer(object):
    def __init__(self, vocab_file=None, segment='jieba'):
        self.words_count = dict()
        if vocab_file is not None:
            self.vocab = utils.read_lines(vocab_file)
            utils.verbose('loading vocab from file {} with vocab_size {}'.format(
                vocab_file, self.vocab_size))
        else:
            self.vocab = []
        self.sub_cutter = SubCutter(chinese_seg=segment)
        self.vocab_dict = dict()
        self.build_vocab_dict()
        self.PAD_ID = 0

    def build_vocab_dict(self):
        self.vocab_dict = {w: i for i, w in enumerate(self.vocab)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def collect_vocab(self, lines):
        words_count = dict()
        for n, line in enumerate(lines, start=1):
            if not n % 10:
                utils.verbose('processing no.{} lines'.format(n))
            tokens = self.sub_cutter.cut(line)
            for token in tokens:
                if token in words_count:
                    words_count[token] += 1
                elif token.startswith('{{') and token.endswith('}}'):
                    new_token = '<' + token.split(':')[0][2:] + '>'
                    if new_token in words_count:
                        words_count[new_token] += 1
                    else:
                        words_count[new_token] = 1
                else:
                    words_count[token] = 1
        words_count = sorted(words_count, key=words_count.get, reverse=True)
        return words_count

    def _build_vocab(self, data, vocab_size):
        self.words_count = self.collect_vocab(data)
        self.vocab = copy_head + list(self.words_count)
        if len(self.vocab) > vocab_size:
            self.vocab = self.vocab[: vocab_size]
        utils.verbose('real vocab: {}, final vocab: {}'.format(
            len(self.words_count), self.vocab_size))
        self.build_vocab_dict()

    def build_vocab(self, data, vocab_size, path):
        self._build_vocab(data, vocab_size)
        utils.write_lines(path, self.vocab)
        utils.verbose('vocab has been dumped in {}'.format(os.path.abspath(path)))

    def token_to_id(self, token):
        if token.startswith('{{') and token.endswith('}}'):
            token = '<' + token.split(':')[0][2:] + '>'
        idx = self.vocab_dict.get(token, self.vocab_dict[UNK])
        return idx

    def tokens_to_ids(self, tokens):
        return [self.token_to_id(token) for token in tokens]

    def encode_line(self, line):
        tokens = self.sub_cutter.cut(line)
        return self.tokens_to_ids(tokens)
