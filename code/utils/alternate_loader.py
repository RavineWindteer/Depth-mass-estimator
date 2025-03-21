import random

def alternate_loader(loader1, loader2):
    # Create iterators for each loader
    iter1 = iter(loader1)
    iter2 = iter(loader2)

    # List to keep track of remaining iterators
    iters = [iter1, iter2]

    while iters:
        # Randomly select an iterator
        iter_idx = random.randint(0, len(iters) - 1)
        selected_iter = iters[iter_idx]

        try:
            # Yield the next batch from the selected iterator
            yield next(selected_iter)
        except StopIteration:
            # If the selected iterator is exhausted, remove it from the list
            iters.remove(selected_iter)
