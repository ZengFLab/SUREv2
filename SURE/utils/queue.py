import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = 0  # unique sequence count

    def is_empty(self):
        return not self.elements

    def put(self, item, priority):
        if item in self.entry_finder:
            self.remove(item)  # Remove the existing entry if it exists
        entry = [priority, self.counter, item]  # New entry with priority, counter, and item
        self.entry_finder[item] = entry
        heapq.heappush(self.elements, entry)
        self.counter += 1

    def remove(self, item):
        # Mark an existing task as REMOVED
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED  # Mark it as removed

    def get(self):
        while self.elements:
            priority, count, item = heapq.heappop(self.elements)
            if item is not self.REMOVED:
                del self.entry_finder[item]  # Remove from the entry finder
                return item
        raise KeyError('pop from an empty priority queue')

    def peek(self):
        while self.elements:
            priority, count, item = self.elements[0]
            if item is not self.REMOVED:
                return item
            heapq.heappop(self.elements)  # Remove stale entry
        return None

    def update(self, item, priority):
        self.put(item, priority)  # Use put to add or update the item

    def __iter__(self):
        # Iterate over the priority queue without popping items
        # This will create a shallow copy of the current elements
        temp_elements = [(priority, count, item) for priority, count, item in self.elements if item is not self.REMOVED]
        for priority, count, item in sorted(temp_elements):
            yield item
