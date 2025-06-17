import torch, random

class CurriculumSampler(torch.utils.data.Sampler):
    def __init__(self, grades, epoch, total_epochs):
        self.grades = grades
        self.epoch = epoch
        self.total_epochs = total_epochs

        max_grade = int((epoch / (total_epochs - 1)) * 12)
        self.allowed = [i for i, g in enumerate(grades) if g <= max_grade]

    def __iter__(self):
        random.shuffle(self.allowed)
        return iter(self.allowed)

    def __len__(self):
        return len(self.allowed)
