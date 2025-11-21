import sys

def write_matrix_txt(filename, m, n):
    with open(filename, "w") as f:
        f.write(f"{m} {n}\n")
        for _ in range(m):
            row = " ".join(["1"] * n)
            f.write(row + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_matrix.py <m> <n>")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    fname = sys.argv[3]

    write_matrix_txt(fname, m, n)
    print(f"Wrote matrix.txt with dimensions {m}x{n}")
