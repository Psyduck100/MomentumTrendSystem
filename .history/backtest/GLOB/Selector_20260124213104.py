class Selector:
    def select(self, date, prices, returns, universe):
        raise NotImplementedError

class EqualWeightSelector(Selector):
    def select(self, date, prices, returns, universe):
        universe = list(universe)

        if len(universe) == 0:
            return {}
        
        w= 1.0/len(universe)
        return {t: w for t in universe}