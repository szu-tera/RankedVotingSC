from collections import Counter

# Base class for rank-based voting systems
class RankBasedSC:
    def __init__(self, resps):
        self.resps = [resp for resp in resps if "invalid" not in resp]

    def run(self):
        raise NotImplementedError

# Instant Runoff Voting (IRV)
class IRV(RankBasedSC):
    def run(self):
        if not self.resps:
            return ""

        def get_winner(votes):
            count = Counter(votes)
            total_votes = len(votes)
            for candidate, num_votes in count.items():
                if num_votes > total_votes / 2:
                    return candidate
            return None

        def eliminate_last_place(votes):
            count = Counter(votes)
            last_place = count.most_common()[-1][0]
            return [vote for vote in votes if vote != last_place]

        votes = [resp[0] for resp in self.resps]
        while True:
            winner = get_winner(votes)
            if winner:
                return winner
            votes = eliminate_last_place(votes)

# Borda Count Voting (BCV)
class BCV(RankBasedSC):
    def run(self):
        if not self.resps:
            return ""

        scores = Counter()
        for resp in self.resps:
            for i, candidate in enumerate(resp):
                scores[candidate] += len(resp) - i
        return scores.most_common(1)[0][0]

# Mean Reciprocal Rank Voting (MRRV)
class MRRV(RankBasedSC):
    def run(self):
        if not self.resps:
            return ""

        scores = Counter()
        for resp in self.resps:
            for rank, option in enumerate(resp, start=1):
                scores[option] += 1 / rank
        return scores.most_common(1)[0][0]