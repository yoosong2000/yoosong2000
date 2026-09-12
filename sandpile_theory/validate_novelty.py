"""Novelty-seeking experiments in pure Python.

Same algorithm as TheorySandpileNovelty.jl, for cross-checking.
"""
import random, math
from collections import defaultdict

OFFS = ((1, 0), (-1, 0), (0, 1), (0, -1))

class ModelNovelty:
    def __init__(self, L=24, threshold=4, credit_slots=3, credit_decay=0.5,
                 novelty_weight=0.5, novelty_bonus=1.0, sample=5, seed=42):
        self.L = L
        self.threshold = threshold
        self.credit_slots = credit_slots
        self.credit_decay = credit_decay
        self.novelty_weight = novelty_weight
        self.novelty_bonus = novelty_bonus
        self.sample = sample
        self.rng = random.Random(seed)
        self.occ = defaultdict(list)
        self.visits = defaultdict(int)
        self.pos = {}
        self.credit = {}
        self.moves = {}
        self.next_id = 1
        self.clock = 0
        self.created = 0
        self.avalanches = []

    def inside(self, p):
        return 1 <= p[0] <= self.L and 1 <= p[1] <= self.L

    def rand_pos(self):
        return (self.rng.randint(1, self.L), self.rng.randint(1, self.L))

    def n_occ(self, p):
        return len(self.occ[p])

    def developed(self, p):
        return self.visits[p] > 0

    def empty_neighbors(self, p):
        return [q for d in OFFS if self.inside(q := (p[0]+d[0], p[1]+d[1]))
                and not self.developed(q)]

    def choose_pos(self):
        cands = [self.rand_pos() for _ in range(self.sample)]
        
        def score(p):
            occ = self.n_occ(p)
            vis = self.visits[p]
            if self.novelty_weight < 0.5:
                return (occ, vis)
            else:
                return (0 if vis == 0 else 100, occ)
        
        return min(cands, key=score)

    def arrive(self, sid, p):
        v = self.visits[p]
        credit_earned = 0.0
        if v < self.credit_slots:
            credit_earned = self.credit_decay ** v
            if v == 0 and self.novelty_weight > 0:
                credit_earned += self.novelty_weight * self.novelty_bonus
        self.credit[sid] += credit_earned
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
        movers = self.occ[p][-self.threshold:]
        del self.occ[p][-self.threshold:]
        dirs = list(OFFS)
        self.rng.shuffle(dirs)
        for sid, d in zip(movers, dirs):
            q = (p[0] + d[0], p[1] + d[1])
            self.moves[sid] += 1
            if self.inside(q):
                self.arrive(sid, q)
            else:
                del self.pos[sid], self.credit[sid], self.moves[sid]
        
        # Theory creation
        if self.novelty_weight > 0.3:
            for q in self.empty_neighbors(p):
                if self.rng.random() < self.novelty_weight * 0.1:
                    self.visits[q] = 1
                    self.created += 1

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

def run(nw, L=24, warm=10000, meas=30000, seed=1):
    m = ModelNovelty(L=L, novelty_weight=nw, seed=seed)
    for _ in range(warm):
        m.step(record=False)
    for _ in range(meas):
        m.step(record=True)
    sizes = [a[0] for a in m.avalanches if a[0] > 0]
    dens = sum(len(m.occ[i, j]) for i in range(1, L + 1)
               for j in range(1, L + 1)) / L ** 2
    b = logbin(sizes)
    sl = slope(b, 4, L * L / 4) if b else float("nan")
    return dens, len(sizes) / meas, sum(sizes) / max(len(sizes), 1), -sl, m.created

if __name__ == "__main__":
    print("\n############  Novelty-seeking sweep (L=24)  ############\n")
    print("novelty_w  density  activity   <s>    τ_fit  theories")
    print("─" * 58)
    for nw in [x/10 for x in range(11)]:
        dens, act, mean_s, tau, created = run(nw)
        print(f"    {nw:.1f}    {dens:.3f}    {act:.3f}   {mean_s:6.1f}  {tau:.3f}      {created}")
    print("─" * 58)
    print("Watch how τ changes with novelty-seeking weight")
