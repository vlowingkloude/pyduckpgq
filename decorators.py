import functools
from uuid import uuid1
import numpy as np

def require_rowid(f):
    """ The first arg must be the table"""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        table = args[1]
        try:
            from pandas.core.frame import DataFrame as pddf
        except (ImportError, OSError) as e:
            class pddf:
                pass
        try:
            from polars.dataframe.frame import DataFrame as pldf
        except (ImportError, OSError) as e:
            class pldf:
                pass
        try:
            from pyarrow.lib import Table as patab
            import pyarrow as pa
        except (ImportError, OSError) as e:
            class patab:
                pass
        def pddf_deleter(df, original = None):
            df.drop("rowid", axis = 1, inplace = True)
            if original:
                df.rename(columns = {original: "rowid"}, inplace = True)
            return df
        def pldf_deleter(df, original = None):
            df = df.drop("rowid")
            if original:
                df = df.rename({original: "rowid"})
            return df
        def numpy_deleter(df, original = None):
            df.pop("rowid")
            if original:
                df["rowid"] = df[original]
                df.pop(original)
            return df
        def patab_deleter(df, original):
            df = df.drop("rowid")
            if original:
                newnames = df.column_names
                newnames[newnames.index(original)] = "rowid"
                df = df.rename_columns(newnames)
            return df

        def get_newname_for_rowid(l):
            i = 0
            while True:
                if f"rowid_{i}" not in l:
                    return i
                i += 1
        if isinstance(table, pddf):
            original = None
            if 'rowid' in table.columns:
                original = f"rowid_{get_newname_for_rowid(table.columns)}"
                table.rename(columns = {"rowid": original}, inplace = True)
            table['rowid'] = table.index
            deleter = pddf_deleter
            f(*args, **kwargs)
        elif isinstance(table, pldf):
            original = None
            if 'rowid' in table.columns:
                original = f"rowid_{get_newname_for_rowid(table.columns)}"
                table = table.rename({"rowid": original})
            table = table.with_row_count("rowid")
            deleter = pldf_deleter
        elif isinstance(table, dict):
            original = None
            if 'rowid' in table.keys():
                original = f"rowid_{get_newname_for_rowid(table.keys())}"
                table[original] = table['rowid']
            table['rowid'] = np.arange(len(table[list(table.keys())[0]]))
            deleter = numpy_deleter
        elif isinstance(table, patab):
            original = None
            if 'rowid' in table.column_names:
                original = f"rowid_{get_newname_for_rowid(table.columns)}"
                newnames = table.column_names
                newnames[newnames.index("rowid")] = original
                table = table.rename_columns(newnames)
            table = table.append_column("rowid", pa.array(np.arange(table.num_rows)))
            deleter = patab_deleter
        elif isinstance(table, str):
            original = None
            deleter = None
        else:
            raise Exception("Not supported type of table")
        res = f(*args, **kwargs)
        if deleter:
            table = deleter(table, original)
        return res
    return wrapper
