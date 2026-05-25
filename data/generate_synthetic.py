import random, json, os
from pathlib import Path
import pandas as pd

random.seed(42)

CXX_SAMPLES = [
    ("Write a C++ function that adds two numbers", "int add(int a, int b) { return a + b; }"),
    ("Write a C++ function that checks if a number is even", "bool is_even(int n) { return n % 2 == 0; }"),
    ("Write a C++ function that returns the maximum of two integers", "int max(int a, int b) { return (a > b) ? a : b; }"),
    ("Implement a C++ function to compute factorial recursively", "int factorial(int n) { return (n <= 1) ? 1 : n * factorial(n - 1); }"),
    ("Write a C++ function to reverse a string", """#include <string>
#include <algorithm>
std::string reverse_string(const std::string& s) {
    std::string r = s;
    std::reverse(r.begin(), r.end());
    return r;
}"""),
    ("Implement a C++ function to find the length of a string", "int string_length(const std::string& s) { return s.length(); }"),
    ("Write a C++ lambda to square a number", "auto square = [](int x) { return x * x; };"),
    ("Implement a C++ function to swap two integers", "void swap(int& a, int& b) { int t = a; a = b; b = t; }"),
    ("Write a C++ function to check if a character is a vowel", "bool is_vowel(char c) { c = tolower(c); return c=='a'||c=='e'||c=='i'||c=='o'||c=='u'; }"),
    ("Implement a C++ function to sum elements in a vector", """#include <vector>
#include <numeric>
int sum_vector(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 0);
}"""),
    ("Write a C++ class for a simple 2D point", """class Point {
public:
    Point(double x, double y) : x_(x), y_(y) {}
    double x() const { return x_; }
    double y() const { return y_; }
    double distance_to(const Point& other) const;
private:
    double x_, y_;
};"""),
    ("Implement a C++ function to print a vector", """#include <iostream>
#include <vector>
template<typename T>
void print_vector(const std::vector<T>& v) {
    for (const auto& e : v) std::cout << e << " ";
    std::cout << std::endl;
}"""),
    ("Write a C++ function that reads a file into a string", """#include <string>
#include <fstream>
#include <sstream>
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}"""),
    ("Implement a C++ function to count occurrences in a vector", """#include <vector>
#include <algorithm>
int count_occurrences(const std::vector<int>& v, int target) {
    return std::count(v.begin(), v.end(), target);
}"""),
    ("Write a C++ function that checks if a string is palindrome", """#include <string>
#include <algorithm>
bool is_palindrome(const std::string& s) {
    std::string rev = s;
    std::reverse(rev.begin(), rev.end());
    return s == rev;
}"""),
    ("Implement a C++ function to generate Fibonacci numbers", """#include <vector>
std::vector<int> fibonacci(int n) {
    std::vector<int> fib = {0, 1};
    for (int i = 2; i < n; ++i)
        fib.push_back(fib[i-1] + fib[i-2]);
    return fib;
}"""),
    ("Write a C++ function to find an element in a vector", """#include <vector>
#include <algorithm>
bool contains(const std::vector<int>& v, int x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}"""),
    ("Implement a C++ function for linear search", """#include <vector>
int linear_search(const std::vector<int>& v, int target) {
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i] == target) return i;
    return -1;
}"""),
    ("Write a C++ RAII wrapper for a mutex lock", """class LockGuard {
public:
    explicit LockGuard(std::mutex& m) : mtx_(m) { mtx_.lock(); }
    ~LockGuard() { mtx_.unlock(); }
private:
    std::mutex& mtx_;
};"""),
    ("Implement a C++ function to concatenate two vectors", """#include <vector>
std::vector<int> concat(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> r = a;
    r.insert(r.end(), b.begin(), b.end());
    return r;
}"""),
]

LUA_SAMPLES = [
    ("Write a Lua function that adds two numbers", "function add(a, b) return a + b end"),
    ("Write a Lua function to check if a number is even", "function is_even(n) return n % 2 == 0 end"),
    ("Implement a Lua function that returns the maximum of two values", "function max(a, b) return (a > b) and a or b end"),
    ("Write a Lua function to compute factorial", "function factorial(n) if n <= 1 then return 1 else return n * factorial(n - 1) end end"),
    ("Write a Lua function to reverse a table", """function reverse_table(t)
    local rev = {}
    for i = #t, 1, -1 do
        table.insert(rev, t[i])
    end
    return rev
end"""),
    ("Implement a Lua function to get the length of a string", "function string_length(s) return #s end"),
    ("Write a Lua function to swap two values", "function swap(a, b) return b, a end"),
    ("Implement a Lua class using metatables for a 2D vector", """Vector = {}
function Vector:new(x, y)
    local obj = {x = x, y = y}
    setmetatable(obj, self)
    self.__index = self
    return obj
end
function Vector:add(other)
    return Vector:new(self.x + other.x, self.y + other.y)
end"""),
    ("Write a Lua function to read a file line by line", """function read_lines(filename)
    local lines = {}
    local file = io.open(filename, "r")
    if not file then return nil end
    for line in file:lines() do
        table.insert(lines, line)
    end
    file:close()
    return lines
end"""),
    ("Implement a Lua function to check if a value is in a table", """function table_contains(t, value)
    for _, v in ipairs(t) do
        if v == value then return true end
    end
    return false
end"""),
    ("Write a Lua function to merge two tables", """function merge_tables(t1, t2)
    local result = {}
    for k, v in pairs(t1) do result[k] = v end
    for k, v in pairs(t2) do result[k] = v end
    return result
end"""),
    ("Write a Lua iterator that yields even numbers", """function even_numbers(limit)
    local i = 0
    return function()
        i = i + 2
        if i > limit then return nil end
        return i
    end
end"""),
    ("Implement a Lua function to serialize a table to a string", """function serialize(t, indent)
    indent = indent or 0
    local parts = {}
    local pad = string.rep("  ", indent)
    for k, v in pairs(t) do
        if type(v) == "table" then
            table.insert(parts, pad .. k .. " = {\\n" .. serialize(v, indent+1) .. pad .. "}")
        else
            table.insert(parts, pad .. k .. " = " .. tostring(v))
        end
    end
    return table.concat(parts, ",\\n")
end"""),
    ("Write a Lua function that deep-copies a table", """function deep_copy(t)
    if type(t) ~= "table" then return t end
    local copy = {}
    for k, v in pairs(t) do
        copy[deep_copy(k)] = deep_copy(v)
    end
    return copy
end"""),
    ("Implement a Lua coroutine-based producer", """function producer()
    return coroutine.create(function()
        for i = 1, 10 do
            coroutine.yield(i)
        end
    end)
end"""),
    ("Write a Lua function to filter a table", """function filter(t, pred)
    local result = {}
    for _, v in ipairs(t) do
        if pred(v) then table.insert(result, v) end
    end
    return result
end"""),
    ("Implement a Lua function to map over a table", """function map(t, fn)
    local result = {}
    for i, v in ipairs(t) do
        result[i] = fn(v)
    end
    return result
end"""),
    ("Write a Lua script that implements a simple event dispatcher", """local EventDispatcher = {}
EventDispatcher.__index = EventDispatcher

function EventDispatcher:new()
    return setmetatable({listeners = {}}, self)
end

function EventDispatcher:on(event, callback)
    if not self.listeners[event] then
        self.listeners[event] = {}
    end
    table.insert(self.listeners[event], callback)
end

function EventDispatcher:emit(event, ...)
    if self.listeners[event] then
        for _, cb in ipairs(self.listeners[event]) do
            cb(...)
        end
    end
end"""),
    ("Write a Lua module with math utilities", """local math_utils = {}

function math_utils.clamp(val, min, max)
    return math.max(min, math.min(max, val))
end

function math_utils.lerp(a, b, t)
    return a + (b - a) * t
end

return math_utils"""),
    ("Implement a Lua function to split a string", """function split(str, sep)
    sep = sep or ","
    local result = {}
    for match in (str .. sep):gmatch("(.-)" .. sep) do
        table.insert(result, match)
    end
    return result
end"""),
]

TEMPLATES = [
    "Write a {language} function that {task}",
    "Implement a {language} class for {task}",
    "Create a {language} program that {task}",
]

def main():
    out_dir = Path(__file__).parent / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for lang, samples in [("C++", CXX_SAMPLES), ("Lua", LUA_SAMPLES)]:
        for (instr, code) in samples:
            fmt = random.choice(TEMPLATES)
            instruction = fmt.format(language=lang, task=instr.lower().replace("write a " if lang == "C++" else "write a ", "").replace("implement a " if lang == "C++" else "implement a ", "").replace("a " + lang.lower() + " function that ", "").replace("a " + lang.lower() + " class for ", ""))
            text = (
                f"### Instruction\n{instruction}\n\n"
                f"### Response\n```{lang.lower()}\n{code}\n```"
            )
            records.append({"text": text, "language": lang, "instruction": instruction})

    random.shuffle(records)

    split = int(len(records) * 0.95)
    train_df = pd.DataFrame(records[:split])
    eval_df = pd.DataFrame(records[split:])

    train_df.to_parquet(out_dir / "train_instructions.parquet", index=False)
    eval_df.to_parquet(out_dir / "eval_instructions.parquet", index=False)

    print(f"Generated {len(records)} instruction pairs ({len(train_df)} train, {len(eval_df)} eval)")
    lang_dist = pd.Series([r["language"] for r in records]).value_counts().to_dict()
    print(f"Language distribution: {lang_dist}")


if __name__ == "__main__":
    main()
