class BinningHelper:
    @staticmethod
    def bin_age(age):
        if age <= 12:
            return "Children"
        elif age <= 20:
            return "Teen"
        elif age <= 39:
            return "Young Adult"
        elif age <= 59:
            return "Middle Age"
        else:
            return "Senior"

    @staticmethod
    def bin_cholesterol(chol):
        if chol <= 199:
            return "Low"
        elif chol <= 239:
            return "Borderline High"
        else:
            return "High"

    @staticmethod
    def index_sleep(hours):
        if hours < 6:
            return 0  # poor
        elif 6 <= hours <= 8:
            return 1  # healthy
        else:
            return 2  # oversleep
    