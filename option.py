import numpy as np 

def intersec(row, col, ratio):
    matrix_result = []
    matrix_area = []
    for r in row:
        # if not tm.is_valid:tm = make_valid(tm)
        mat = []
        area = []
        for c in col:
            # if not pm.is_valid:pm = make_valid(pm)
            intersec = r.intersection(c)
            if intersec.is_empty:
                mat.append(0)
                area.append(0)
            elif intersec:
                intersec_area = intersec.area
                area.append(intersec_area)
                table_area = r.area
                person_area = c.area
                smaller_poly_area = min(table_area, person_area)
                if intersec_area >= ratio * smaller_poly_area:
                    mat.append(1)
                else:mat.append(0)
            else: 
                mat.append(0)
                area.append(0)
        matrix_result.append(mat)
        matrix_area.append(area)
    return np.array(matrix_result, dtype=np.int32), np.array(matrix_area, dtype=np.float64)