# Python3 gdb pretty-printers for bits of SuiteSparse

import gdb
import re
from bisect import bisect_left

class _EntryIterator(object):

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.row = -1    # so first call to "next" gives [0,0]
        self.col = 0

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()   # for Python 2.7

    def __next__(self):
        self.row = self.row + 1
        if self.row >= self.rows:
            self.row = 0
            self.col = self.col + 1

        if self.col >= self.cols:
            raise StopIteration

        return (self.row, self.col)

class CSparsePrinter:
    "Print a cholmod_sparse matrix"

    def __init__(self, val):

        type = val.type
        if type.code == gdb.TYPE_CODE_REF:
                type = type.target()
        self.type = type.unqualified().strip_typedefs()
        tag = self.type.tag

        self.val = val

    class _iterator(_EntryIterator):
        def __init__(self, rows, cols, val, dataPtr):
            super(CSparsePrinter._iterator, self).__init__(rows, cols)
            self.val = val
            self.dataPtr = dataPtr

        def __next__(self):
            row, col = super(CSparsePrinter._iterator, self).__next__()

            # get stored value
            # assuming sorted and packed with doubles and long indices

            # index into columns to find starting point
            long_ptr = gdb.lookup_type("long").pointer()
            p = self.val['p'].cast(long_ptr)
            colstart = p + col      # offset of first row entry (both idx and value)
            colend   = p + col + 1  # offset of first row entry of next col

            # p is void* (yay, C polymorphism) so handle them appropriately
            colstart = colstart.cast(long_ptr).dereference()
            colend   = colend.cast(long_ptr).dereference()
            nz       = colend - colstart

            # is our row index in the half-open range i[p[col]] .. i[p[col+1]] ?
            i_array = self.val['i'].cast(long_ptr)
            row_indices = [(i_array + x).dereference() for x in range(int(colstart), int(colend))]
            row_ofs = bisect_left(row_indices, row)
            if row_ofs != nz:
                # found our row index; access the value
                double_ptr = gdb.lookup_type("double").pointer()
                data = self.val['x'].cast(double_ptr)
                return ('[%d,%d]'%(row, col), (data + colstart + row_ofs).dereference())
            else:
                return ('[%d,%d]'%(row, col), gdb.Value(0.0))

    def children(self):
        return self._iterator(self.val['nrow'], self.val['ncol'], self.val, None)

    def to_string(self):

        if self.val['packed'] != 0:
            status = "packed"
        else:
            status = "unpacked"

        if self.val['sorted'] !=0:
            status = status + " sorted"
        else:
            status = status + " unsorted"

        dimensions  = "%d x %d" % (self.val['nrow'], self.val['ncol'])

        itype = "CHOLMOD_INT"
        if self.val['itype'] == 1:
            itype = "CHOLMOD_INTLONG"
        elif self.val['itype'] == 2:
            itype = "CHOLMOD_LONG"

        if self.val['xtype'] == 0:
            xtype = "pattern"
        elif self.val['xtype'] == 1:
            xtype = "real"
        elif self.val['xtype'] == 2:
            xtype = "complex"
        elif self.val['xtype'] == 3:
            xtype = "zomplex"   # MATLAB complex
            
        if self.val['dtype'] == 0:
            dtype = 'double'
        else:
            dtype = 'float'

        return "cholmod_sparse, %s, %s, %s" % (dimensions, status, dtype)

    @staticmethod
    def supported(val):
        # if we cannot print this variant (yet), do not respond to the matcher
        # pretty restrictive for the moment
        if val['stype'] != 0:
            # symmetric with upper or lower triangular data
            return False

        if val['itype'] != 2:   # CHOLMOD_LONG
            return False

        if val['xtype'] != 1:   # CHOLMOD_REAL
            return False

        if val['dtype'] != 0:   # CHOLMOD_DOUBLE
            return False

        return True

def pp_lookup(val):
    "find a pretty-printer that can print the supplied value"

    typ = val.type
    if typ.code == gdb.TYPE_CODE_REF:    # strip references
        typ = typ.target()

    typ = typ.unqualified().strip_typedefs()

    typ = typ.tag

    if typ is None:
        return None

    if re.match('^cholmod_sparse', typ) and CSparsePrinter.supported(val):
        return CSparsePrinter(val)

    return None

def register_cholmod_printers(obj = None):
    "Register cholmod pretty-printers with objfile Obj"

    if obj is None:
        obj = gdb
    obj.pretty_printers.append(pp_lookup)
