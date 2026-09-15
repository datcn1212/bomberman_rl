### Constants read from `settings.py`

| name | value |
|---|---|
| `COLS` | 17 |
| `ROWS` | 17 |
| `MAX_AGENTS` | 4 |
| `MAX_STEPS` | 400 |
| `BOMB_POWER` | 3 |
| `BOMB_TIMER` | 4 |
| `EXPLOSION_TIMER` | 2 |
| `TIMEOUT` | 0.5 |
| `TRAIN_TIMEOUT` | inf |
| `REWARD_KILL` | 5 |
| `REWARD_COIN` | 1 |
| scenario `empty` | CRATE_DENSITY 0, COIN_COUNT 0 |
| scenario `coin-heaven` | CRATE_DENSITY 0, COIN_COUNT 50 |
| scenario `loot-crate` | CRATE_DENSITY 0.75, COIN_COUNT 50 |
| scenario `classic` | CRATE_DENSITY 0.75, COIN_COUNT 9 |

**A. Drop bomb at spawn, step out of the blast, watch the full life cycle**

| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |
|---|---|---|---|---|---|
| 1 | BOMB | (1, 1) | True | - | - |
| 2 | RIGHT | (1, 1) | False | (1,1)t=3 | - |
| 3 | RIGHT | (2, 1) | False | (1,1)t=2 | - |
| 4 | DOWN | (3, 1) | False | (1,1)t=1 | - |
| 5 | WAIT | (3, 2) | False | (1,1)t=0 | - |
| 6 | WAIT | (3, 2) | False | - | (1,1)=1 (1,2)=1 (1,3)=1 (1,4)=1 (2,1)=1 (3,1)=1 (4,1)=1 |
| 7 | WAIT | (3, 2) | False | - | - |
| 8 | WAIT | (3, 2) | True | - | - |
| 9 | WAIT | (3, 2) | True | - | - |

**A. Measured**

- `BOMB` issued at step 1 from (1, 1); the bomb becomes visible one step later with timer 3 and counts down [3, 2, 1, 0].
- The agent survives the whole round, so the timer sequence above is complete: the blast lands at the end of the step where the observed timer is 0 (step 5), i.e. **4 steps after the BOMB action** (`BOMB_TIMER` = 4).
- `explosion_map` shows dangerous cells on steps [6] with values [1].
- `bombs_left` goes False at step 2 and back to True at step 8 (**7 steps after the BOMB action**), independently of where the agent is standing.

**B. Blast geometry, bomb at (1, 1) (open corner)**

- Burning cells: [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]
- Straight-line model with walls predicts: [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)]
- Match: **True**
- Off-axis cells (a blast that turned a corner would show up here): **none -> blasts never turn corners**

**C. Bomb next to interior pillars (tests wall blocking)**

| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |
|---|---|---|---|---|---|
| 1 | DOWN | (1, 1) | True | - | - |
| 2 | DOWN | (1, 2) | True | - | - |
| 3 | RIGHT | (1, 3) | True | - | - |
| 4 | BOMB | (2, 3) | True | - | - |
| 5 | LEFT | (2, 3) | False | (2,3)t=3 | - |
| 6 | DOWN | (1, 3) | False | (2,3)t=2 | - |
| 7 | WAIT | (1, 4) | False | (2,3)t=1 | - |
| 8 | WAIT | (1, 4) | False | (2,3)t=0 | - |
| 9 | WAIT | (1, 4) | False | - | (1,3)=1 (2,3)=1 (3,3)=1 (4,3)=1 (5,3)=1 |
| 10 | WAIT | (1, 4) | False | - | - |
| 11 | WAIT | (1, 4) | True | - | - |
| 12 | WAIT | (1, 4) | True | - | - |
| 13 | WAIT | (1, 4) | True | - | - |
| 14 | WAIT | (1, 4) | True | - | - |
| 15 | WAIT | (1, 4) | True | - | - |
| 16 | WAIT | (1, 4) | True | - | - |
| 17 | WAIT | (1, 4) | True | - | - |
| 18 | WAIT | (1, 4) | True | - | - |
| 19 | WAIT | (1, 4) | True | - | - |
| 20 | WAIT | (1, 4) | True | - | - |
| 21 | WAIT | (1, 4) | True | - | - |
| 22 | WAIT | (1, 4) | True | - | - |
| 23 | WAIT | (1, 4) | True | - | - |
| 24 | WAIT | (1, 4) | True | - | - |
| 25 | WAIT | (1, 4) | True | - | - |
| 26 | WAIT | (1, 4) | True | - | - |
| 27 | WAIT | (1, 4) | True | - | - |
| 28 | WAIT | (1, 4) | True | - | - |
| 29 | WAIT | (1, 4) | True | - | - |
| 30 | WAIT | (1, 4) | True | - | - |
| 31 | WAIT | (1, 4) | True | - | - |
| 32 | WAIT | (1, 4) | True | - | - |
| 33 | WAIT | (1, 4) | True | - | - |
| 34 | WAIT | (1, 4) | True | - | - |
| 35 | WAIT | (1, 4) | True | - | - |
| 36 | WAIT | (1, 4) | True | - | - |
| 37 | WAIT | (1, 4) | True | - | - |
| 38 | WAIT | (1, 4) | True | - | - |
| 39 | WAIT | (1, 4) | True | - | - |
| 40 | WAIT | (1, 4) | True | - | - |
| 41 | WAIT | (1, 4) | True | - | - |
| 42 | WAIT | (1, 4) | True | - | - |
| 43 | WAIT | (1, 4) | True | - | - |
| 44 | WAIT | (1, 4) | True | - | - |
| 45 | WAIT | (1, 4) | True | - | - |
| 46 | WAIT | (1, 4) | True | - | - |
| 47 | WAIT | (1, 4) | True | - | - |
| 48 | WAIT | (1, 4) | True | - | - |
| 49 | WAIT | (1, 4) | True | - | - |
| 50 | WAIT | (1, 4) | True | - | - |
| 51 | WAIT | (1, 4) | True | - | - |
| 52 | WAIT | (1, 4) | True | - | - |
| 53 | WAIT | (1, 4) | True | - | - |
| 54 | WAIT | (1, 4) | True | - | - |
| 55 | WAIT | (1, 4) | True | - | - |
| 56 | WAIT | (1, 4) | True | - | - |
| 57 | WAIT | (1, 4) | True | - | - |
| 58 | WAIT | (1, 4) | True | - | - |
| 59 | WAIT | (1, 4) | True | - | - |
| 60 | WAIT | (1, 4) | True | - | - |
| 61 | WAIT | (1, 4) | True | - | - |
| 62 | WAIT | (1, 4) | True | - | - |
| 63 | WAIT | (1, 4) | True | - | - |
| 64 | WAIT | (1, 4) | True | - | - |
| 65 | WAIT | (1, 4) | True | - | - |
| 66 | WAIT | (1, 4) | True | - | - |
| 67 | WAIT | (1, 4) | True | - | - |
| 68 | WAIT | (1, 4) | True | - | - |
| 69 | WAIT | (1, 4) | True | - | - |
| 70 | WAIT | (1, 4) | True | - | - |
| 71 | WAIT | (1, 4) | True | - | - |
| 72 | WAIT | (1, 4) | True | - | - |
| 73 | WAIT | (1, 4) | True | - | - |
| 74 | WAIT | (1, 4) | True | - | - |
| 75 | WAIT | (1, 4) | True | - | - |
| 76 | WAIT | (1, 4) | True | - | - |
| 77 | WAIT | (1, 4) | True | - | - |
| 78 | WAIT | (1, 4) | True | - | - |
| 79 | WAIT | (1, 4) | True | - | - |
| 80 | WAIT | (1, 4) | True | - | - |
| 81 | WAIT | (1, 4) | True | - | - |
| 82 | WAIT | (1, 4) | True | - | - |
| 83 | WAIT | (1, 4) | True | - | - |
| 84 | WAIT | (1, 4) | True | - | - |
| 85 | WAIT | (1, 4) | True | - | - |
| 86 | WAIT | (1, 4) | True | - | - |
| 87 | WAIT | (1, 4) | True | - | - |
| 88 | WAIT | (1, 4) | True | - | - |
| 89 | WAIT | (1, 4) | True | - | - |
| 90 | WAIT | (1, 4) | True | - | - |
| 91 | WAIT | (1, 4) | True | - | - |
| 92 | WAIT | (1, 4) | True | - | - |
| 93 | WAIT | (1, 4) | True | - | - |
| 94 | WAIT | (1, 4) | True | - | - |
| 95 | WAIT | (1, 4) | True | - | - |
| 96 | WAIT | (1, 4) | True | - | - |
| 97 | WAIT | (1, 4) | True | - | - |
| 98 | WAIT | (1, 4) | True | - | - |
| 99 | WAIT | (1, 4) | True | - | - |
| 100 | WAIT | (1, 4) | True | - | - |
| 101 | WAIT | (1, 4) | True | - | - |
| 102 | WAIT | (1, 4) | True | - | - |
| 103 | WAIT | (1, 4) | True | - | - |
| 104 | WAIT | (1, 4) | True | - | - |
| 105 | WAIT | (1, 4) | True | - | - |
| 106 | WAIT | (1, 4) | True | - | - |
| 107 | WAIT | (1, 4) | True | - | - |
| 108 | WAIT | (1, 4) | True | - | - |
| 109 | WAIT | (1, 4) | True | - | - |
| 110 | WAIT | (1, 4) | True | - | - |
| 111 | WAIT | (1, 4) | True | - | - |
| 112 | WAIT | (1, 4) | True | - | - |
| 113 | WAIT | (1, 4) | True | - | - |
| 114 | WAIT | (1, 4) | True | - | - |
| 115 | WAIT | (1, 4) | True | - | - |
| 116 | WAIT | (1, 4) | True | - | - |
| 117 | WAIT | (1, 4) | True | - | - |
| 118 | WAIT | (1, 4) | True | - | - |
| 119 | WAIT | (1, 4) | True | - | - |
| 120 | WAIT | (1, 4) | True | - | - |
| 121 | WAIT | (1, 4) | True | - | - |
| 122 | WAIT | (1, 4) | True | - | - |
| 123 | WAIT | (1, 4) | True | - | - |
| 124 | WAIT | (1, 4) | True | - | - |
| 125 | WAIT | (1, 4) | True | - | - |
| 126 | WAIT | (1, 4) | True | - | - |
| 127 | WAIT | (1, 4) | True | - | - |
| 128 | WAIT | (1, 4) | True | - | - |
| 129 | WAIT | (1, 4) | True | - | - |
| 130 | WAIT | (1, 4) | True | - | - |
| 131 | WAIT | (1, 4) | True | - | - |
| 132 | WAIT | (1, 4) | True | - | - |
| 133 | WAIT | (1, 4) | True | - | - |
| 134 | WAIT | (1, 4) | True | - | - |
| 135 | WAIT | (1, 4) | True | - | - |
| 136 | WAIT | (1, 4) | True | - | - |
| 137 | WAIT | (1, 4) | True | - | - |
| 138 | WAIT | (1, 4) | True | - | - |
| 139 | WAIT | (1, 4) | True | - | - |
| 140 | WAIT | (1, 4) | True | - | - |
| 141 | WAIT | (1, 4) | True | - | - |
| 142 | WAIT | (1, 4) | True | - | - |
| 143 | WAIT | (1, 4) | True | - | - |
| 144 | WAIT | (1, 4) | True | - | - |
| 145 | WAIT | (1, 4) | True | - | - |
| 146 | WAIT | (1, 4) | True | - | - |
| 147 | WAIT | (1, 4) | True | - | - |
| 148 | WAIT | (1, 4) | True | - | - |
| 149 | WAIT | (1, 4) | True | - | - |
| 150 | WAIT | (1, 4) | True | - | - |
| 151 | WAIT | (1, 4) | True | - | - |
| 152 | WAIT | (1, 4) | True | - | - |
| 153 | WAIT | (1, 4) | True | - | - |
| 154 | WAIT | (1, 4) | True | - | - |
| 155 | WAIT | (1, 4) | True | - | - |
| 156 | WAIT | (1, 4) | True | - | - |
| 157 | WAIT | (1, 4) | True | - | - |
| 158 | WAIT | (1, 4) | True | - | - |
| 159 | WAIT | (1, 4) | True | - | - |
| 160 | WAIT | (1, 4) | True | - | - |
| 161 | WAIT | (1, 4) | True | - | - |
| 162 | WAIT | (1, 4) | True | - | - |
| 163 | WAIT | (1, 4) | True | - | - |
| 164 | WAIT | (1, 4) | True | - | - |
| 165 | WAIT | (1, 4) | True | - | - |
| 166 | WAIT | (1, 4) | True | - | - |
| 167 | WAIT | (1, 4) | True | - | - |
| 168 | WAIT | (1, 4) | True | - | - |
| 169 | WAIT | (1, 4) | True | - | - |
| 170 | WAIT | (1, 4) | True | - | - |
| 171 | WAIT | (1, 4) | True | - | - |
| 172 | WAIT | (1, 4) | True | - | - |
| 173 | WAIT | (1, 4) | True | - | - |
| 174 | WAIT | (1, 4) | True | - | - |
| 175 | WAIT | (1, 4) | True | - | - |
| 176 | WAIT | (1, 4) | True | - | - |
| 177 | WAIT | (1, 4) | True | - | - |
| 178 | WAIT | (1, 4) | True | - | - |
| 179 | WAIT | (1, 4) | True | - | - |
| 180 | WAIT | (1, 4) | True | - | - |
| 181 | WAIT | (1, 4) | True | - | - |
| 182 | WAIT | (1, 4) | True | - | - |
| 183 | WAIT | (1, 4) | True | - | - |
| 184 | WAIT | (1, 4) | True | - | - |
| 185 | WAIT | (1, 4) | True | - | - |
| 186 | WAIT | (1, 4) | True | - | - |
| 187 | WAIT | (1, 4) | True | - | - |
| 188 | WAIT | (1, 4) | True | - | - |
| 189 | WAIT | (1, 4) | True | - | - |
| 190 | WAIT | (1, 4) | True | - | - |
| 191 | WAIT | (1, 4) | True | - | - |
| 192 | WAIT | (1, 4) | True | - | - |
| 193 | WAIT | (1, 4) | True | - | - |
| 194 | WAIT | (1, 4) | True | - | - |
| 195 | WAIT | (1, 4) | True | - | - |
| 196 | WAIT | (1, 4) | True | - | - |
| 197 | WAIT | (1, 4) | True | - | - |
| 198 | WAIT | (1, 4) | True | - | - |
| 199 | WAIT | (1, 4) | True | - | - |
| 200 | WAIT | (1, 4) | True | - | - |
| 201 | WAIT | (1, 4) | True | - | - |
| 202 | WAIT | (1, 4) | True | - | - |
| 203 | WAIT | (1, 4) | True | - | - |
| 204 | WAIT | (1, 4) | True | - | - |
| 205 | WAIT | (1, 4) | True | - | - |
| 206 | WAIT | (1, 4) | True | - | - |
| 207 | WAIT | (1, 4) | True | - | - |
| 208 | WAIT | (1, 4) | True | - | - |
| 209 | WAIT | (1, 4) | True | - | - |
| 210 | WAIT | (1, 4) | True | - | - |
| 211 | WAIT | (1, 4) | True | - | - |
| 212 | WAIT | (1, 4) | True | - | - |
| 213 | WAIT | (1, 4) | True | - | - |
| 214 | WAIT | (1, 4) | True | - | - |
| 215 | WAIT | (1, 4) | True | - | - |
| 216 | WAIT | (1, 4) | True | - | - |
| 217 | WAIT | (1, 4) | True | - | - |
| 218 | WAIT | (1, 4) | True | - | - |
| 219 | WAIT | (1, 4) | True | - | - |
| 220 | WAIT | (1, 4) | True | - | - |
| 221 | WAIT | (1, 4) | True | - | - |
| 222 | WAIT | (1, 4) | True | - | - |
| 223 | WAIT | (1, 4) | True | - | - |
| 224 | WAIT | (1, 4) | True | - | - |
| 225 | WAIT | (1, 4) | True | - | - |
| 226 | WAIT | (1, 4) | True | - | - |
| 227 | WAIT | (1, 4) | True | - | - |
| 228 | WAIT | (1, 4) | True | - | - |
| 229 | WAIT | (1, 4) | True | - | - |
| 230 | WAIT | (1, 4) | True | - | - |
| 231 | WAIT | (1, 4) | True | - | - |
| 232 | WAIT | (1, 4) | True | - | - |
| 233 | WAIT | (1, 4) | True | - | - |
| 234 | WAIT | (1, 4) | True | - | - |
| 235 | WAIT | (1, 4) | True | - | - |
| 236 | WAIT | (1, 4) | True | - | - |
| 237 | WAIT | (1, 4) | True | - | - |
| 238 | WAIT | (1, 4) | True | - | - |
| 239 | WAIT | (1, 4) | True | - | - |
| 240 | WAIT | (1, 4) | True | - | - |
| 241 | WAIT | (1, 4) | True | - | - |
| 242 | WAIT | (1, 4) | True | - | - |
| 243 | WAIT | (1, 4) | True | - | - |
| 244 | WAIT | (1, 4) | True | - | - |
| 245 | WAIT | (1, 4) | True | - | - |
| 246 | WAIT | (1, 4) | True | - | - |
| 247 | WAIT | (1, 4) | True | - | - |
| 248 | WAIT | (1, 4) | True | - | - |
| 249 | WAIT | (1, 4) | True | - | - |
| 250 | WAIT | (1, 4) | True | - | - |
| 251 | WAIT | (1, 4) | True | - | - |
| 252 | WAIT | (1, 4) | True | - | - |
| 253 | WAIT | (1, 4) | True | - | - |
| 254 | WAIT | (1, 4) | True | - | - |
| 255 | WAIT | (1, 4) | True | - | - |
| 256 | WAIT | (1, 4) | True | - | - |
| 257 | WAIT | (1, 4) | True | - | - |
| 258 | WAIT | (1, 4) | True | - | - |
| 259 | WAIT | (1, 4) | True | - | - |
| 260 | WAIT | (1, 4) | True | - | - |
| 261 | WAIT | (1, 4) | True | - | - |
| 262 | WAIT | (1, 4) | True | - | - |
| 263 | WAIT | (1, 4) | True | - | - |
| 264 | WAIT | (1, 4) | True | - | - |
| 265 | WAIT | (1, 4) | True | - | - |
| 266 | WAIT | (1, 4) | True | - | - |
| 267 | WAIT | (1, 4) | True | - | - |
| 268 | WAIT | (1, 4) | True | - | - |
| 269 | WAIT | (1, 4) | True | - | - |
| 270 | WAIT | (1, 4) | True | - | - |
| 271 | WAIT | (1, 4) | True | - | - |
| 272 | WAIT | (1, 4) | True | - | - |
| 273 | WAIT | (1, 4) | True | - | - |
| 274 | WAIT | (1, 4) | True | - | - |
| 275 | WAIT | (1, 4) | True | - | - |
| 276 | WAIT | (1, 4) | True | - | - |
| 277 | WAIT | (1, 4) | True | - | - |
| 278 | WAIT | (1, 4) | True | - | - |
| 279 | WAIT | (1, 4) | True | - | - |
| 280 | WAIT | (1, 4) | True | - | - |
| 281 | WAIT | (1, 4) | True | - | - |
| 282 | WAIT | (1, 4) | True | - | - |
| 283 | WAIT | (1, 4) | True | - | - |
| 284 | WAIT | (1, 4) | True | - | - |
| 285 | WAIT | (1, 4) | True | - | - |
| 286 | WAIT | (1, 4) | True | - | - |
| 287 | WAIT | (1, 4) | True | - | - |
| 288 | WAIT | (1, 4) | True | - | - |
| 289 | WAIT | (1, 4) | True | - | - |
| 290 | WAIT | (1, 4) | True | - | - |
| 291 | WAIT | (1, 4) | True | - | - |
| 292 | WAIT | (1, 4) | True | - | - |
| 293 | WAIT | (1, 4) | True | - | - |
| 294 | WAIT | (1, 4) | True | - | - |
| 295 | WAIT | (1, 4) | True | - | - |
| 296 | WAIT | (1, 4) | True | - | - |
| 297 | WAIT | (1, 4) | True | - | - |
| 298 | WAIT | (1, 4) | True | - | - |
| 299 | WAIT | (1, 4) | True | - | - |
| 300 | WAIT | (1, 4) | True | - | - |
| 301 | WAIT | (1, 4) | True | - | - |
| 302 | WAIT | (1, 4) | True | - | - |
| 303 | WAIT | (1, 4) | True | - | - |
| 304 | WAIT | (1, 4) | True | - | - |
| 305 | WAIT | (1, 4) | True | - | - |
| 306 | WAIT | (1, 4) | True | - | - |
| 307 | WAIT | (1, 4) | True | - | - |
| 308 | WAIT | (1, 4) | True | - | - |
| 309 | WAIT | (1, 4) | True | - | - |
| 310 | WAIT | (1, 4) | True | - | - |
| 311 | WAIT | (1, 4) | True | - | - |
| 312 | WAIT | (1, 4) | True | - | - |
| 313 | WAIT | (1, 4) | True | - | - |
| 314 | WAIT | (1, 4) | True | - | - |
| 315 | WAIT | (1, 4) | True | - | - |
| 316 | WAIT | (1, 4) | True | - | - |
| 317 | WAIT | (1, 4) | True | - | - |
| 318 | WAIT | (1, 4) | True | - | - |
| 319 | WAIT | (1, 4) | True | - | - |
| 320 | WAIT | (1, 4) | True | - | - |
| 321 | WAIT | (1, 4) | True | - | - |
| 322 | WAIT | (1, 4) | True | - | - |
| 323 | WAIT | (1, 4) | True | - | - |
| 324 | WAIT | (1, 4) | True | - | - |
| 325 | WAIT | (1, 4) | True | - | - |
| 326 | WAIT | (1, 4) | True | - | - |
| 327 | WAIT | (1, 4) | True | - | - |
| 328 | WAIT | (1, 4) | True | - | - |
| 329 | WAIT | (1, 4) | True | - | - |
| 330 | WAIT | (1, 4) | True | - | - |
| 331 | WAIT | (1, 4) | True | - | - |
| 332 | WAIT | (1, 4) | True | - | - |
| 333 | WAIT | (1, 4) | True | - | - |
| 334 | WAIT | (1, 4) | True | - | - |
| 335 | WAIT | (1, 4) | True | - | - |
| 336 | WAIT | (1, 4) | True | - | - |
| 337 | WAIT | (1, 4) | True | - | - |
| 338 | WAIT | (1, 4) | True | - | - |
| 339 | WAIT | (1, 4) | True | - | - |
| 340 | WAIT | (1, 4) | True | - | - |
| 341 | WAIT | (1, 4) | True | - | - |
| 342 | WAIT | (1, 4) | True | - | - |
| 343 | WAIT | (1, 4) | True | - | - |
| 344 | WAIT | (1, 4) | True | - | - |
| 345 | WAIT | (1, 4) | True | - | - |
| 346 | WAIT | (1, 4) | True | - | - |
| 347 | WAIT | (1, 4) | True | - | - |
| 348 | WAIT | (1, 4) | True | - | - |
| 349 | WAIT | (1, 4) | True | - | - |
| 350 | WAIT | (1, 4) | True | - | - |
| 351 | WAIT | (1, 4) | True | - | - |
| 352 | WAIT | (1, 4) | True | - | - |
| 353 | WAIT | (1, 4) | True | - | - |
| 354 | WAIT | (1, 4) | True | - | - |
| 355 | WAIT | (1, 4) | True | - | - |
| 356 | WAIT | (1, 4) | True | - | - |
| 357 | WAIT | (1, 4) | True | - | - |
| 358 | WAIT | (1, 4) | True | - | - |
| 359 | WAIT | (1, 4) | True | - | - |
| 360 | WAIT | (1, 4) | True | - | - |
| 361 | WAIT | (1, 4) | True | - | - |
| 362 | WAIT | (1, 4) | True | - | - |
| 363 | WAIT | (1, 4) | True | - | - |
| 364 | WAIT | (1, 4) | True | - | - |
| 365 | WAIT | (1, 4) | True | - | - |
| 366 | WAIT | (1, 4) | True | - | - |
| 367 | WAIT | (1, 4) | True | - | - |
| 368 | WAIT | (1, 4) | True | - | - |
| 369 | WAIT | (1, 4) | True | - | - |
| 370 | WAIT | (1, 4) | True | - | - |
| 371 | WAIT | (1, 4) | True | - | - |
| 372 | WAIT | (1, 4) | True | - | - |
| 373 | WAIT | (1, 4) | True | - | - |
| 374 | WAIT | (1, 4) | True | - | - |
| 375 | WAIT | (1, 4) | True | - | - |
| 376 | WAIT | (1, 4) | True | - | - |
| 377 | WAIT | (1, 4) | True | - | - |
| 378 | WAIT | (1, 4) | True | - | - |
| 379 | WAIT | (1, 4) | True | - | - |
| 380 | WAIT | (1, 4) | True | - | - |
| 381 | WAIT | (1, 4) | True | - | - |
| 382 | WAIT | (1, 4) | True | - | - |
| 383 | WAIT | (1, 4) | True | - | - |
| 384 | WAIT | (1, 4) | True | - | - |
| 385 | WAIT | (1, 4) | True | - | - |
| 386 | WAIT | (1, 4) | True | - | - |
| 387 | WAIT | (1, 4) | True | - | - |
| 388 | WAIT | (1, 4) | True | - | - |
| 389 | WAIT | (1, 4) | True | - | - |
| 390 | WAIT | (1, 4) | True | - | - |
| 391 | WAIT | (1, 4) | True | - | - |
| 392 | WAIT | (1, 4) | True | - | - |
| 393 | WAIT | (1, 4) | True | - | - |
| 394 | WAIT | (1, 4) | True | - | - |
| 395 | WAIT | (1, 4) | True | - | - |
| 396 | WAIT | (1, 4) | True | - | - |
| 397 | WAIT | (1, 4) | True | - | - |
| 398 | WAIT | (1, 4) | True | - | - |
| 399 | WAIT | (1, 4) | True | - | - |
| 400 | WAIT | (1, 4) | True | - | - |

- Bomb at (2, 3); neighbouring walls: [(2, 2), (2, 4)]
- Burning cells: [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]
- Straight-line-with-walls prediction: [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]
- Match: **True** -> a wall stops the ray at the wall, the blast does not continue past it.

**D. Drop a bomb and stand still**

| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |
|---|---|---|---|---|---|
| 1 | BOMB | (1, 1) | True | - | - |
| 2 | WAIT | (1, 1) | False | (1,1)t=3 | - |
| 3 | WAIT | (1, 1) | False | (1,1)t=2 | - |
| 4 | WAIT | (1, 1) | False | (1,1)t=1 | - |
| 5 | WAIT | (1, 1) | False | (1,1)t=0 | - |

- The agent is polled for the last time at step 5 (observed bomb timer 0), so it dies during that step: **a bomb whose observed timer is 0 kills at the end of the current step**, leaving no further chance to move.

**E-early. Escape, then step back into the blast row**

| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |
|---|---|---|---|---|---|
| 1 | BOMB | (1, 1) | True | - | - |
| 2 | RIGHT | (1, 1) | False | (1,1)t=3 | - |
| 3 | RIGHT | (2, 1) | False | (1,1)t=2 | - |
| 4 | DOWN | (3, 1) | False | (1,1)t=1 | - |
| 5 | UP | (3, 2) | False | (1,1)t=0 | - |

- Step back attempted at step 5; explosion cells observed then: []; last polled step 5 -> DIED

**E-late. Escape, then step back into the blast row**

| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |
|---|---|---|---|---|---|
| 1 | BOMB | (1, 1) | True | - | - |
| 2 | RIGHT | (1, 1) | False | (1,1)t=3 | - |
| 3 | RIGHT | (2, 1) | False | (1,1)t=2 | - |
| 4 | DOWN | (3, 1) | False | (1,1)t=1 | - |
| 5 | WAIT | (3, 2) | False | (1,1)t=0 | - |
| 6 | UP | (3, 2) | False | - | (1,1)=1 (1,2)=1 (1,3)=1 (1,4)=1 (2,1)=1 (3,1)=1 (4,1)=1 |

- Step back attempted at step 6; explosion cells observed then: [[1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 4, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]]; last polled step 6 -> DIED

**E-later. Escape, then step back into the blast row**

| step | act | pos | bombs_left | bombs (pos,timer) | explosion_map dangerous cells |
|---|---|---|---|---|---|
| 1 | BOMB | (1, 1) | True | - | - |
| 2 | RIGHT | (1, 1) | False | (1,1)t=3 | - |
| 3 | RIGHT | (2, 1) | False | (1,1)t=2 | - |
| 4 | DOWN | (3, 1) | False | (1,1)t=1 | - |
| 5 | WAIT | (3, 2) | False | (1,1)t=0 | - |
| 6 | WAIT | (3, 2) | False | - | (1,1)=1 (1,2)=1 (1,3)=1 (1,4)=1 (2,1)=1 (3,1)=1 (4,1)=1 |
| 7 | UP | (3, 2) | False | - | - |
| 8 | WAIT | (3, 1) | True | - | - |
| 9 | WAIT | (3, 1) | True | - | - |

- Step back attempted at step 7; explosion cells observed then: []; last polled step 9 -> SURVIVED
