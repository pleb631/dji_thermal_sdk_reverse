"""IIRP emissivity calibration."""

from base64 import b64decode
from bisect import bisect_left, bisect_right
from struct import Struct, unpack
from zlib import decompress


_FLOAT = Struct("<f")
_EMISSIVITIES = tuple(value / 1000 for value in range(800, 1000, 2))
_TEMPERATURES = tuple(
    value / 100 for value in (*range(22315, 82316, 1000), *range(87315, 182316, 5000))
)
_COLUMNS = len(_EMISSIVITIES)
_LEGACY_EMISSIVITIES = tuple(value / 100 for value in range(80, 100))
_LEGACY_TEMPERATURES = tuple(value / 100 for value in range(22315, 107316, 1000))
# ems_correct_table_v1 from libv_iirp.so, rounded to the output float32 format.
_LEGACY_TABLE = (
    b"eNrt2LFrU1EUBvDMmTNKgwgSKUjhIYgQThEkpRQkRRAhCCI8BEFSRJCWjpkz65rMmTOna50zZ27XdravcOHk5Dvn"
    b"nvveizj0H/hxe77vnBfaaOzTfOcNnXQP6c/lET0bvKXV6TvqNt/T718f6LYzoFnvEzWXnynPc7q4+ULt0Vc6a32j"
    b"5WRI48UPuu7/pIPVKU2H59Sa9das0VV/zTqef9zwgpVl3zc8acm3FZ72Ns2zLOtvRV7sbQ/eg6f1uW4PvbFuj7+x"
    b"bs9zD8J9Sc0k5qXcq/A3e73wRk/G0pO3uUzG0vPMsIwX67XHG4xfJ2WidZB71gxRJh7PmmEVr3ijJ2PNa9x9z9EM"
    b"U70iE+6hGcY6E/OKN8pMrM4gbzrs3ptWJlYHNU/O0OqM17N+Z3m9g9Urc4Yo45hXvDHFk51GnpVJinfdfxnNBHXQ"
    b"8uQMrYzRjiCPzxBljDqIvPHihZkJ6kzM4zO0Mi7j8UxSveJmBY9nonmo0/IbkmWZK+MUD2Xyv3jLyV40Y9Tpur2w"
    b"c8hDmfCM6/ZCZ7SbwL2z1nOYidXpMp7MGO2I5slM6vYa4n8AvIMxrz3aXcsEdaaKFzJGHeQ7wm+M9EImqDNVvJBx"
    b"rNPIC7/ZLm46rg5KL+yw5oWM6/ZQB+XOxbyQsdfjN8vyUAfljsS8PH9a2QvfOO6hDsodkTfG4/EOlvVQB+WOyB3W"
    b"vObyCeyg1ws3P+bxTssd9ni8g3V7vNNy59A3RHpaB7fh8Z3TbjT3Zr3HsNNyR7Sb+q883mm5c9aNRh7vINo57QZW"
    b"8eSN0bzj+Y65c+gmeD25I9vw5E3werzT2g5X8eTOpXi801W9286jDQ/dhLKe3Lkqnty5bXvyZiHvL8K3mBI="
)
# ems_correct_table_v2 from libv_iirp.so, in units of 0.0001.
# The payload stores signed differences along both table axes.
_TABLE = (
    b"eNqNWzuPHEUQripL/ACCs40xD/M7+EdI/B1CIjIiAhISJAICRAASEkjm1mf7zNkSAUgYMN5iZ3t7uh5f9eyt5tXT"
    b"013vrkffx/cf0AN6j94/HO/SO4ff/eNxn94+/u4dj3v01vHov7vrcZfuHM/L9c7pPH63zfX28dev7e7icL44Pl2c"
    b"WsfThTu3u3GNvz/opf6lf+vLw/GP/ns4Xh3O/x7Or/S/49F+r0/H/njdH47226uerv6n5tyu7W+/npd2ot7e37aW"
    b"cSYab0Z779/biD6iAcNrM/M+wNHnfX38bm9mbTOM2dXd6QqPhZGPd2yg6uf9etfvW//x1N+O3svxIQ0qzXH33w1q"
    b"qKqDpmHVZ2pPTP3QE/x8ardfjS/6Ha399dh39OjPveWDtW1AwyvOFro4W8eAAwT5u/4VJzj1SOkxLoNZbF919LEj"
    b"vGl6Mu3Nu/06ToQQPY+Z+li+hx+xtzQ6xPm5eFYHkZ//DeKAm59nQFiN40fk03l/GsuO1FqWN7xiYN/YFj8Gh3nY"
    b"zWZlw1Mw/rGZLcI26M/mDUM8B1wWEzQfgt7C0XHpfV4GakQaVbN02FUF0sz3beMx4A7qN8NNDE/t35+K6G6liAP0"
    b"XoIHBgxxtdTJ7zlQO8/LQC4iLMvf7ycKcDmPlToOUobh89h5SfCtdsw4M+qRqdJ7vFB2MsFJH7xM9zOSMV61hDff"
    b"YYw5QYFaIhTL/Y1miCIMnePk4IqzIS7VPS11It/H3AykYWiK/eJ6nbu/idS0rUhX/HwVVdoYMhlH1j7IRrLhYeYH"
    b"0VPF+I3reON5FUfOEpP5EDV1xqtqHj9ff/tYLUXZSausfM+05STXvs3zKsLBQCsY8jbq1ugvTjKvlNPaE2kV5YSB"
    b"nuZv2NDef+31iKHN5jP6WypfKRXSkOUS4dXpQgXWDPV9PlLmM17J7Jtd0k+k03ndiBKXe1PZZrnI4K7LD7Yw+Iud"
    b"RvyyXuNRhpfABR1r2du2BUgOonTaPjulwipUePnVQYLd9HeII5WkZCg5aUzsM+h4qVgn4mhMtR5x0oDcxlN7MBuH"
    b"S52w151Wdm9GT2zVKwxoYq/Y2R0OXEaSON5avv+qmL7b0pFXRyxlmP6WWhJsqJ8/0klMFDDwvVS0So91sH+Xda6/"
    b"58nK6z0zBnaRJ5ZaklwO7sUZLoGvi1dZKnUeS1T2weL6nC2VBD8b+UHYn73UOEs1wrY/OPPBrFzEVVaCf4/9bqzr"
    b"/W6nVH6HdbseE/myWdOjDeRSo3NvdpFE93vE8cPGGVxGD/lNlNex7tQ2M+IvQNs8NByildGjzXypDKCKfryk+JA3"
    b"OFjbZdns4X224evKxDfepeg5ruE1zDNJ89qc472aJoiXKHrwkrHT6DF1W4vjFxSZVZJ2js+MJA+vg9FWe8v4SKuM"
    b"QG5Hnt2WvzKTN+wHVTzw3OWwBjzSys8kt9oKyI+IyzvhlZQAZ1FsiK0Y9mwQ1juV5CHGmfP6VNlI5KVFL4PKiKKP"
    b"5mN4KfwBP/eVomxNva7TGfmevCovOWcpokVvTa395/ScfeauB1dKgLPY453J/MwGY+vDxt9jEDtRKVFIFh5p5vg5"
    b"KwNNcwGe84guOOeT/cOcjxPowzxWHOtVOYeYX+QyN5Bzel5f9g6iysPNEXRc1dtcj1MVhYvokJL3XPHN5wswZ73V"
    b"xDkxv0Lg7MnIw1UZPJRPrXJOPloRl1c5z2PBMsrA88c6Ovgha6aeQZYUcabyT73k+myRlQ+U+4oaUGl7fPtE2cA/"
    b"JDdiUdUTcH6CkseG/DGUMY1ajqPDSKklTx1tg612bkUXUXZxRqj2qxAf8Mrl430xkUf7e6K1Pc30oiKjySDbGHMR"
    b"NktYZeJQtFh7ybb1WmMVFekWzgj02jDDvDiqEWCIkDTimL+2gE+nK1TEIdcgbgXtqqwUxmaeXTkvU9zurzXXK/Ya"
    b"Y87ZM0+psPW0lVNAskQp76N6rfUKhdfkaFmr+i6V6yWu+TBVldlYoUHW+Jn6Ki9D64RyDigCypYYVTBr723mAXBZ"
    b"Y1n+nmnl3yrM0OGKQvZoIyd9tiJ6IbxRMyaY57VzDn4IqFZ6mRx7OhANKwuLdAnpyHbcF+uD1po8UwkyhfZgRIua"
    b"q6GoujzLsKCKHNr1sO2NtuNG/c4ZrL05Q5Zr0rnWXle1cn28yu/P6ll2jt8UVT2yPPj9SUTnrPN1NJPjAF69Ok67"
    b"QFD1MMJyo3hHUoTb7zOoYeZyV0+mks+o+IoI3pNQ7za4SbqBMoqU+MXQZuG8VtZ4bEHyXoXsM6Mq4nK+CZXdZc69"
    b"MtivUlPdS+aAOmdUvK2z8VKk8zyLk+3B87CXT9z+PLTHrpYkvKOIYV4g2wJkFaWMdCMMz7XaQ1athX5f5CxHZCMF"
    b"3lgvcv2fyuwrur7QrAVxlUCWh0DWM7dUu8LqXT4o37iVQVvOn6zRsKy48yq1/ionCjONXrKuz2J6+fv+3L8U931s"
    b"49Qm5o2kb+R4fKodtlYpsRVdj0lrHfZD1tqkxcvDR6eZyPThADE52Hj9UlLPQQ1ZR+24fHaq8wyq9sqPmMoVOYoP"
    b"myMmLrDcYQf5gIkCbI0yGENy8Hq42++WGfvzNV73lloMTyI+FOjfPSWG0No+kvCjxK3ICStH5Lhle34Ba/w2N0sJ"
    b"G/+WV8w5fcvpG3GYZaxtizgdY6OhXueWpy817nBgkJGI/GJXtfQc8b3FYWUx7d6UhO88PlJIqJdWoa/KfMpoE1Dv"
    b"YlPLw1wckknAAlrrccvM5LXQ88bKoQTMv1aCnMBZorwvh11VM1qK8S7KW8SZII7RpuR1oH/5jc6isFhb9OurhBy+"
    b"pByuJL9Ugn3rvSVpHAe7TlStbcsI32rMYnrvUEAszakWFqtoXj4ZSF3Wtghrl6IhuQLtf2v9Ts/Le+V9FAz3D/re"
    b"EvTOY+htQG8hJ2loRSZg975XXzOQafwmweu09oDhDqN8L5BbFrsGhaz5XmzFvVb9oFTkdOYZTS53u8qUoxYXSViP"
    b"Wn2EHbeO0X6EPug86+33ZKM9CFEWBezpG9yspY+dV4HX3jbOT4qj/1x3ULNi4nxM3E/Ua05o13LtQRBcZwfcfWX2"
    b"fPxZqch5VDtN8M41nGUTWIWud1xIoTN5DaXgQ/yiuEJWZRUY7q/Y2qOV/7cj2wi7Ho04aOgFBYtvVy+mh6o6i9Rm"
    b"makcg1YV81qHrF3KO+Cypyom72G/eQjyOxlCnL1vuZWMicAaBa7AcrFTJu7GQlJmV+MdiM9HlF79F9DsP3jm/1FD"
    b"xd5YBrVcb3Np6gv8D/bPvhQ="
)


def _float32(value):
    return _FLOAT.unpack(_FLOAT.pack(value))[0]


def _load_table():
    cells = []
    for index, delta in enumerate(unpack("<8100h", decompress(b64decode(_TABLE)))):
        row, column = divmod(index, _COLUMNS)
        left = cells[index - 1] if column else 0
        above = cells[index - _COLUMNS] if row else 0
        diagonal = cells[index - _COLUMNS - 1] if row and column else 0
        cells.append(delta + left + above - diagonal)
    return tuple(_float32(value / 10000) for value in cells)


_CELLS = _load_table()
_LEGACY_CELLS = unpack("<1720f", decompress(b64decode(_LEGACY_TABLE)))
_LEGACY_TEMPERATURE_INDEX = tuple(map(_float32, _LEGACY_TEMPERATURES))
_LEGACY_EMISSIVITY_INDEX = tuple(map(_float32, _LEGACY_EMISSIVITIES))
# Both SDK bin searches advance past a cell within 0.0001 of its axis value.
_TEMPERATURE_INDEX = tuple(value - 0.0001 for value in _TEMPERATURES)
_EMISSIVITY_INDEX = tuple(value - 0.0001 for value in _EMISSIVITIES)


def _nearest(axis, value):
    upper = bisect_right(axis, value)
    if upper == 0:
        return 0
    if upper == len(axis) or value - axis[upper - 1] < axis[upper] - value:
        return upper - 1
    return upper


def lookup_emissivity_v1(apparent_celsius, emissivity):
    """Look up the nearest legacy temperature and emissivity bins."""
    emissivity = _float32(emissivity)
    if not _LEGACY_EMISSIVITIES[0] <= emissivity <= _LEGACY_EMISSIVITIES[-1]:
        return emissivity
    kelvin = _float32(_float32(apparent_celsius) + 273.15)
    row = _nearest(_LEGACY_TEMPERATURE_INDEX, kelvin)
    column = _nearest(_LEGACY_EMISSIVITY_INDEX, emissivity)
    return _LEGACY_CELLS[row * len(_LEGACY_EMISSIVITIES) + column]


def lookup_emissivity(apparent_celsius, emissivity, version):
    """Apply the selected IIRP emissivity calibration."""
    emissivity = _float32(emissivity)
    if not 0 < emissivity <= 1:
        raise ValueError("Emissivity must be greater than zero and at most one")
    if version == 0:
        return emissivity
    if version == 1:
        return lookup_emissivity_v1(apparent_celsius, emissivity)
    if version not in (2, 3):
        raise ValueError("Unsupported IIRP emissivity calibration")

    kelvin = _float32(_float32(apparent_celsius) + 273.15)
    row = max(0, bisect_left(_TEMPERATURE_INDEX, kelvin) - 1)
    column = max(0, bisect_left(_EMISSIVITY_INDEX, emissivity) - 1)
    first, last = _EMISSIVITIES[0], _EMISSIVITIES[-1]
    if version == 2 and not first <= emissivity <= last:
        return emissivity
    if emissivity < first:
        edge = _interpolate_column(kelvin, row, 0)
        return _float32(emissivity * (edge / first))
    if emissivity > last:
        edge = _interpolate_column(kelvin, row, _COLUMNS - 1)
        slope = _float32(1 - edge) / (1 - last)
        return _float32(edge + (emissivity - last) * slope)
    if column == _COLUMNS - 1:
        return _interpolate_column(kelvin, row, column)
    if row == len(_TEMPERATURES) - 1:
        e0, e1 = _EMISSIVITIES[column:column + 2]
        a, b = _CELLS[row * _COLUMNS + column:row * _COLUMNS + column + 2]
        return _float32(a * ((e1 - emissivity) / (e1 - e0)) + b * ((emissivity - e0) / (e1 - e0)))
    t0, t1 = _TEMPERATURES[row:row + 2]
    e0, e1 = _EMISSIVITIES[column:column + 2]
    area = (e1 - e0) * (t1 - t0)
    index = row * _COLUMNS + column
    a, b = _CELLS[index:index + 2]
    c, d = _CELLS[index + _COLUMNS:index + _COLUMNS + 2]
    return _float32(
        a * ((e1 - emissivity) * (t1 - kelvin) / area)
        + b * ((emissivity - e0) * (t1 - kelvin) / area)
        + c * ((e1 - emissivity) * (kelvin - t0) / area)
        + d * ((emissivity - e0) * (kelvin - t0) / area)
    )


def _interpolate_column(kelvin, row, column):
    index = row * _COLUMNS + column
    a = _CELLS[index]
    if row == len(_TEMPERATURES) - 1:
        return a
    t0, t1 = _TEMPERATURES[row:row + 2]
    b = _CELLS[index + _COLUMNS]
    return _float32(a * ((t1 - kelvin) / (t1 - t0)) + b * ((kelvin - t0) / (t1 - t0)))
