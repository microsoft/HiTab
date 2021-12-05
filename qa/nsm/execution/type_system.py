def get_hmt_type_hierarchy():
    type_hierarchy = {
        'list': [],
        'ordered_list': ['list'],
        'string_list': ['list'],
        'num_list': ['ordered_list'],
        'datetime_list': ['ordered_list'],

        'head': [],
        'function': ['head'],
        'operation_function': ['function'],
        'filter_function': ['function'],
        'filter_tree_function': ['filter_function'],
        'filter_level_function': ['filter_function'],
        'aggregation_function': ['operation_function'],
        'return_function': ['operation_function'],

        'header': [],
        'ordered_header': ['header'],
        'string_header': ['header'],
        'num_header': ['ordered_header'],
        'datetime_header': ['ordered_header'],

        'index_name': [],
        'direction_index_name': ['index_name'],  # 'INAME_<LEFT>_0', 'INAME_<TOP>_0'
        'non_direction_index_name': ['index_name'],
        'leaf_index_name': ['non_direction_index_name'],  # index name of leaf level.

        'num': [],
        'int': ['num'],

        'string': [],

        # 'direction': [],  # '<LEFT>' | <'TOP'>

        'region': [],  # Tuple[List, List], i.e. operating region in (left_ids, top_ids) format

        # 'level': [],  # '1' | '2'

        'symbol': [],

        '<ERROR>': [],
    }

    return type_hierarchy


def is_number(obj):
    return isinstance(obj, int) or isinstance(obj, float)


class DateTime(object):
    def __init__(self, year=-1, month=-1, day=-1):
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)

        self.year = year
        self.month = month
        self.day = day

        self._day_repr = 365 * (0 if year == -1 else year) + 30 * (0 if month == -1 else month) + (
            0 if day == -1 else day)

        self._hash = hash((self.year, self.month, self.day))

    @property
    def is_month_only(self):
        return self.year == -1 and self.month != -1 and self.day == -1

    def __hash__(self):
        return self._hash

    @property
    def is_year_only(self):
        return self.year != -1 and self.month == self.day == -1

    def __eq__(self, other):
        if not isinstance(other, DateTime): return False

        if other.is_month_only:
            return self.month == other.month
        elif other.is_year_only:
            return self.year == other.year

        return self._day_repr == other._day_repr

    def __ne__(self, other):
        if not isinstance(other, DateTime): return False

        if other.is_month_only:
            return self.month != other.month
        elif other.is_year_only:
            return self.year != other.year

        return self._day_repr != other._day_repr

    def __gt__(self, other):
        if not isinstance(other, DateTime): return False

        if other.is_month_only:
            return self.month > other.month
        elif other.is_year_only:
            return self.year > other.year

        return self._day_repr > other._day_repr

    def __ge__(self, other):
        if not isinstance(other, DateTime): return False

        if other.is_month_only:
            return self.month >= other.month
        elif other.is_year_only:
            return self.year >= other.year

        return self._day_repr >= other._day_repr

    def __lt__(self, other):
        if not isinstance(other, DateTime): return False

        if other.is_month_only:
            return self.month < other.month
        elif other.is_year_only:
            return self.year < other.year

        return self._day_repr < other._day_repr

    def __le__(self, other):
        if not isinstance(other, DateTime): return False

        if other.is_month_only:
            return self.month <= other.month
        elif other.is_year_only:
            return self.year <= other.year

        return self._day_repr <= other._day_repr

    @staticmethod
    def from_string(date_string):
        # read in values by parsing the input string
        # YYYY-MM-DD
        data = date_string.split('-')
        year = -1 if data[0] in ('xxxx', 'xx') else int(data[0])
        month = -1 if data[1] == 'xx' else int(data[1])
        day = -1 if data[2] == 'xx' else int(data[2])

        return DateTime(year, month, day)

    @property
    def ymd(self):
        return (self.year, self.month, self.day)

    def __str__(self):
        return 'Date(%d,%d,%d)' % (self.year, self.month, self.day)

    __repr__ = __str__
