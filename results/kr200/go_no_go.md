# South Korea go/no-go (docs/RULES.md, Rule KG)

- Source: FinanceData/marcap at commit `d3b703628e5a73112111d8bc43439f950762f67e`; every yearly file matches `scripts/kr_data.sha256` (sha256 of that manifest: `12b24b74dd15cddadeb713606c9da6dba18a380f7f9deab225f6fb0f905ec802`).
- Every number below uses only rows dated before 2024-03-01 (Rule ND).
- Written by `python -m detmaso.data.ingest.kr go-no-go`; the rule was fixed before the data was opened.

**Result: GO, all three checks pass.**

| Check | Result |
|---|---|
| KG1, delisted stocks are present | PASS |
| KG2, the split adjustment comes out right | PASS |
| KG3, suspended days can be identified | PASS |

## KG1, delisted stocks are present: PASS

- Rows used: dated before 2024-03-01 (13,848,822 rows, 5,202 codes).
- KOSPI codes whose last row is before 2023-03-01 (more than one year before 2024-03-01): **1151** (needed: at least 200). A code counts as KOSPI if its last row is labelled KOSPI.
- Of these, codes ending in 0 (common shares): 510. Either count clears the bar.
- Known delistings present with their last traded row in the stated year: **3 of 3** (needed: at least 2).

| Stock | Stated year | Last traded row | Matches |
|---|---|---|---|
| Hanjin Shipping (117930) | 2017 | 2017-03-06 | yes |
| Korea Exchange Bank (004940) | 2013 | 2013-04-02 | yes |
| Samsung C&T (old) (000830) | 2015 | 2015-08-26 | yes |

## KG2, the split adjustment comes out right: PASS

- Factor: 1 on a stock's first row, then multiplied on every row by (previous row's close / base price), base price = close - change.
- Samsung Electronics (005930), 50-for-1: factor on 2018-05-04 / factor on 2018-04-27 (its last traded day in April 2018) = **50.000000** (needed: within 1% of 50). Pass.
- Other splits within 1% of their ratio in the stated month: **3 of 3** (needed: at least 2). Ratio = factor on the month's last row / factor on the last row before the month.

| Stock | Split | Month | Factor ratio over the month | Largest one-day step in the month | Within 1% |
|---|---|---|---|---|---|
| Amorepacific (090430) | 10-for-1 | 2015-05 | 9.997426 | 9.997426 on 2015-05-08 | yes |
| Naver (035420) | 5-for-1 | 2018-10 | 4.992908 | 4.992908 on 2018-10-12 | yes |
| Kakao (035720) | 5-for-1 | 2021-04 | 4.982143 | 4.982143 on 2021-04-15 | yes |

## KG3, suspended days can be identified: PASS

- Converted series of Samsung Electronics (005930): the same code path that writes the Qlib files.

| Day | Expected | Converted close | As expected |
|---|---|---|---|
| 2018-04-27 | a price | 2,650,000 raw | yes |
| 2018-04-30 | no price | missing | yes |
| 2018-05-02 | no price | missing | yes |
| 2018-05-03 | no price | missing | yes |
| 2018-05-04 | a price | 51,900 raw | yes |

## Known gap

Cash dividends are not in this source, so the factor does not include them (Rule KG). Share splits, reverse
splits, rights issues and other changes that move the exchange's base price are in the factor.
