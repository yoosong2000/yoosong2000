"""Dependency-free Python reference for the theory-sandpile rules.

Implements exactly the same algorithm as TheorySandpile.jl, as a cross-check
on the dynamics: if the two disagree on the stationary density or the
avalanche exponent, one of them has a bug.  Also usable where Julia is not
installed.  Slow (pure Python) -- keep L small.

    python3 validate_reference.py
"""
import random, math
from collections import defaultdict

OFFS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Model:
    def __init__(self, L=24, threshold=4, credit_slots=3, credit_decay=0.5,
                 drive=":uniform", sample=5, seed=42):
        self.L = L
        self.threshold = threshold
        self.credit_slots = credit_slots
        self.credit_decay = credit_decay
        self.drive = drive
        self.sample = sample
        self.rng = random.Random(seed)
        self.occ = defaultdict(list)          # pos -> [ids], arrival-ordered
        self.visits = defaultdict(int)        # pos -> cumulative arrivals ever
        self.pos = {}                         # id -> pos
        self.credit = {}
        self.moves = {}
        self.next_id = 1
        self.clock = 0
        self.avalanches = []

    def inside(self, p):
        return 1 <= p[0] <= self.L and 1 <= p[1] <= self.L

    def rand_pos(self):
        return (self.rng.randint(1, self.L), self.rng.randint(1, self.L))

    def n_occ(self, p):
        return len(self.occ[p])

    def developed(self, p):
        return self.visits[p] > 0

    def choose_pos(self):
        if self.drive == ":uniform":
            return self.rand_pos()
        cands = [self.rand_pos() for _ in range(self.sample)]
        if self.drive == ":avoid_crowded":
            return min(cands, key=lambda p: (self.n_occ(p), self.visits[p]))
        if self.drive == ":matthew":
            return max(cands, key=lambda p: (self.n_occ(p), self.visits[p]))
        if self.drive == ":frontier":
            fr = [p for p in cands if not self.developed(p) and
                  any(self.developed((p[0] + d[0], p[1] + d[1]))
                      for d in OFFS if self.inside((p[0] + d[0], p[1] + d[1])))]
            return fr[0] if fr else cands[0]
        raise ValueError(self.drive)

    def arrive(self, sid, p):
        """Place scientist sid on theory p and award priority credit."""
        v = self.visits[p]
        if v < self.credit_slots:
            self.credit[sid] += self.credit_decay ** v
        self.visits[p] = v + 1
        self.occ[p].append(sid)
        self.pos[sid] = p

    def enter(self):
        sid = self.next_id
        self.next_id += 1
        self.credit[sid] = 0.0
        self.moves[sid] = 0
        p = self.choose_pos()
        self.arrive(sid, p)
        return p

    def topple(self, p):
        movers = self.occ[p][-self.threshold:]      # the latest arrivals leave
        del self.occ[p][-self.threshold:]
        dirs = list(OFFS)
        self.rng.shuffle(dirs)
        for sid, d in zip(movers, dirs):
            q = (p[0] + d[0], p[1] + d[1])
            self.moves[sid] += 1
            if self.inside(q):
                self.arrive(sid, q)
            else:                                    # leaves the field
                del self.pos[sid], self.credit[sid], self.moves[sid]

    def relax(self, seed_pos):
        size = duration = 0
        toppled = set()
        radius = 0
        queue = [seed_pos]
        while queue:
            fired = False
            nxt = []
            for p in queue:
                if self.n_occ(p) < self.threshold:
                    continue
                self.topple(p)
                fired = True
                size += 1
                toppled.add(p)
                radius = max(radius, abs(p[0] - seed_pos[0]) + abs(p[1] - seed_pos[1]))
                nxt.append(p)
                for d in OFFS:
                    q = (p[0] + d[0], p[1] + d[1])
                    if self.inside(q):
                        nxt.append(q)
            if fired:
                duration += 1
            queue = list(dict.fromkeys(nxt))
        return size, duration, len(toppled), radius

    def step(self, record=True):
        self.clock += 1
        p = self.enter()
        s, d, a, r = self.relax(p)
        if record:
            self.avalanches.append((s, d, a, r))


def mle_alpha(xs, xmin):
    xs = [x for x in xs if x >= xmin]
    n = len(xs)
    if n < 20:
        return float("nan"), n
    s = sum(math.log(x / (xmin - 0.5)) for x in xs)
    return 1.0 + n / s, n


def logbin(xs, base=1.6):
    xs = [x for x in xs if x > 0]
    if not xs:
        return []
    hi = max(xs)
    edges, e = [1.0], 1.0
    while e < hi:
        e = max(e * base, e + 1)
        edges.append(e)
    counts = [0] * (len(edges) - 1)
    for x in xs:
        lo, hi_i = 0, len(edges) - 1
        while lo < hi_i - 1:
            mid = (lo + hi_i) // 2
            if x >= edges[mid]:
                lo = mid
            else:
                hi_i = mid
        counts[lo] += 1
    n = len(xs)
    out = []
    for i, c in enumerate(counts):
        w = edges[i + 1] - edges[i]
        if c:
            out.append((math.sqrt(edges[i] * edges[i + 1]), c / (w * n)))
    return out


def slope(pts, lo, hi):
    pts = [(x, y) for x, y in pts if lo <= x <= hi]
    n = len(pts)
    if n < 3:
        return float("nan")
    lx = [math.log(x) for x, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def run(drive, L=24, warm=20000, meas=60000, seed=1):
    m = Model(L=L, drive=drive, seed=seed)
    for _ in range(warm):
        m.step(record=False)
    for _ in range(meas):
        m.step(record=True)
    sizes = [a[0] for a in m.avalanches if a[0] > 0]
    durs = [a[1] for a in m.avalanches if a[1] > 0]
    dens = sum(m.n_occ((i, j)) for i in range(1, L + 1)
               for j in range(1, L + 1)) / L ** 2
    tau, n = mle_alpha(sizes, 4)
    b = logbin(sizes)
    sl = slope(b, 4, L * L / 4)
    ncred = [c for c in m.credit.values()]
    devd = sum(1 for i in range(1, L + 1) for j in range(1, L + 1) if m.visits[(i, j)] > 0)
    print(f"\n=== drive={drive}  L={L} ===")
    print(f"  stationary density   : {dens:.3f} scientists/theory   (BTW ref ~2.125)")
    print(f"  active avalanches    : {len(sizes)}/{meas} = {len(sizes)/meas:.3f}")
    print(f"  <s>                  : {sum(sizes)/max(len(sizes),1):.1f}   max s = {max(sizes)}")
    print(f"  <T>                  : {sum(durs)/max(len(durs),1):.1f}   max T = {max(durs)}")
    print(f"  tau (MLE, s>=4)      : {tau:.3f}  (n={n})")
    print(f"  tau (log-bin slope)  : {-sl:.3f}")
    print(f"  theories ever worked : {devd}/{L*L}")
    print(f"  live scientists      : {len(m.pos)}")
    print("  log-binned P(s):")
    for x, y in b[:14]:
        print(f"     s~{x:8.1f}   P={y:.3e}   {'#' * max(0,int(28 + 2.2*math.log10(y)))}")
    return m


if __name__ == "__main__":
    for d in (":uniform", ":avoid_crowded", ":matthew", ":frontier"):
        run(d)
