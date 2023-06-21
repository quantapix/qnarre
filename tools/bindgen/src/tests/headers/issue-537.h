struct AlignedToOne {
  int i;
} __attribute__((packed, aligned(1)));

struct AlignedToTwo {
  int i;
} __attribute__((packed, aligned(2)));

#pragma pack(1)

struct PackedToOne {
  int x;
  int y;
};

#pragma pack()

#pragma pack(2)

struct PackedToTwo {
  int x;
  int y;
};

#pragma pack()
