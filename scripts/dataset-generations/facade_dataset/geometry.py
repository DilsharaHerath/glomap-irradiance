"""Facade extraction adapted from drone-prep-new/building_geometry.py.

Same triangle clustering, outward normals, plane ordering and IDs as the
reference notebook. Input assets are read only; no Mitsuba dependency.
"""
from pathlib import Path
import math
import numpy as np

PLY_TYPES = {'char':'i1','uchar':'u1','int8':'i1','uint8':'u1','short':'i2','ushort':'u2',
             'int16':'i2','uint16':'u2','int':'i4','uint':'u4','int32':'i4','uint32':'u4',
             'float':'f4','float32':'f4','double':'f8','float64':'f8'}


def read_ply(path, faces=False):
    """Read the binary triangle PLY format emitted by this dataset's exporter."""
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY file: {path}')
        fmt = None; elements = []; current = None
        while True:
            raw = f.readline()
            if not raw: raise ValueError(f'Truncated PLY header: {path}')
            line = raw.decode('ascii').split()
            if not line: continue
            if line[0] == 'format': fmt = line[1]
            elif line[0] == 'element':
                current = dict(name=line[1], count=int(line[2]), properties=[]); elements.append(current)
            elif line[0] == 'property': current['properties'].append(line[1:])
            elif line[0] == 'end_header': break
        if fmt not in ('binary_little_endian','binary_big_endian') or not elements or elements[0]['name'] != 'vertex':
            raise ValueError(f'Expected vertex-first binary PLY: {path}')
        endian = '<' if fmt == 'binary_little_endian' else '>'
        props = elements[0]['properties']
        if any(p[0] == 'list' for p in props): raise ValueError('Vertex list properties unsupported')
        dtype = np.dtype([(p[1], endian + PLY_TYPES[p[0]]) for p in props])
        data = np.fromfile(f, dtype=dtype, count=elements[0]['count'])
        if len(data) != elements[0]['count']: raise ValueError(f'Truncated vertices: {path}')
        vertices = np.column_stack([data[k] for k in ('x','y','z')]).astype(np.float64)
        if not np.isfinite(vertices).all(): raise ValueError(f'Nonfinite vertices: {path}')
        if not faces: return vertices
        if len(elements)<2 or elements[1]['name']!='face': raise ValueError('Expected faces after vertices')
        face = elements[1]
        if len(face['properties']) != 1 or face['properties'][0][0] != 'list':
            raise ValueError('Unsupported face properties')
        _, count_type, index_type, _ = face['properties'][0]
        # Mitsuba exporter writes triangles; fail explicitly for different formats.
        dtype = np.dtype([('count',endian+PLY_TYPES[count_type]),('indices',endian+PLY_TYPES[index_type],(3,))])
        triangles = np.fromfile(f,dtype=dtype,count=face['count'])
        if len(triangles)!=face['count'] or np.any(triangles['count']!=3):
            raise ValueError(f'Expected triangle faces: {path}')
        indices = triangles['indices'].astype(np.int64)
        if indices.size and (indices.min()<0 or indices.max()>=len(vertices)): raise ValueError('Invalid triangle indices')
        return vertices, indices


def facade_planes(building, scene_dir, min_area=5.0):
    """Find actual vertical triangle planes, including rotated/recessed facades.

    Group geometric normals and signed distances with small tolerances, then
    retain the exact dominant triangle normal/distance for each group. Report
    residuals so rounded clustering is never mistaken for exact coplanarity.
    """
    center=np.array(building['center']); groups={}
    for part in building['parts']:
        v,f=read_ply(scene_dir/part['filename'],faces=True)
        tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(cross,axis=1)
        keep=length>1e-8;tri=tri[keep];normal=cross[keep]/length[keep,None];area=length[keep]/2
        for t,n,a in zip(tri,normal,area):
            if abs(n[1])>0.02: continue
            # Consistent outward sign for main outside envelope surfaces.
            if np.dot(t.mean(axis=0)-center,n)<0: n=-n
            distance=float(t[0]@n)
            angle=math.degrees(math.atan2(n[0],n[2]))%360
            key=(round(angle)%360,round(distance/0.05))
            groups.setdefault(key,[]).append((t,n,float(a),distance))
    planes=[]
    for items in groups.values():
        area=sum(i[2] for i in items)
        if area<min_area: continue
        dominant=max(items,key=lambda i:i[2]);normal=dominant[1];distance=dominant[3]
        # Enforce a vertical camera sampling plane. Exclude sloped facades here.
        if abs(normal[1])>1e-5: continue
        normal=np.array([normal[0],0,normal[2]]);normal/=np.linalg.norm(normal)
        distance=float(dominant[0][0]@normal)
        right=np.cross([0,1,0],normal)
        pts=np.concatenate([i[0] for i in items]);u=pts@right
        residual=float(np.max(np.abs(pts@normal-distance)))
        if residual>0.06: continue
        side=max(dict(east=np.array([-1,0,0]),west=np.array([1,0,0]),north=np.array([0,0,1]),south=np.array([0,0,-1])),
                 key=lambda s:dict(east=[-1,0,0],west=[1,0,0],north=[0,0,1],south=[0,0,-1])[s]@normal)
        planes.append(dict(normal=normal.tolist(),right=right.tolist(),distance=distance,
            u_min=float(u.min()),u_max=float(u.max()),y_min=float(pts[:,1].min()),y_max=float(pts[:,1].max()),
            area=area,side=side,plane_residual_max=residual))
    planes.sort(key=lambda p:(-p['area'],p['side'],p['distance']))
    for i,p in enumerate(planes):p['id']=f'facade_{i:03d}'
    return planes
