%T 값비싼 질문, 거짓말하는 대답: 문항수 가격이 있는 q진 탐색의 정확 계산과 그 한계
%T2 Costly Questions, Lying Answers: Exact Computation and Its Limits for Priced q-ary Search with an Adversarial Lie Budget
%A 허태혁 (Taehyeok Heo) · 2026-11-21
%ABS
질문마다 값이 다르고, 대답은 가끔 거짓이며, 질문의 폭(몇 가지 답을 주는가)도 선택할 수 있다면, 가장 싼 탐색은 무엇인가. 본 논문은 후보 n개 중 하나를 맞히는 문제에서 (i) 질문의 문항수 k에 가격 p_k가 붙고, (ii) 응답자가 전체에서 최대 e번 임의로 거짓말을 할 수 있으며, (iii) 목적이 최악의 경우 총비용 최소화인 모형을 다룬다. 이 모형은 Rényi–Ulam 거짓말 탐색(가격 없음, 이진 답)과 그룹테스트(문항수 고정, 비용 균일)를 양 끝으로 포함한다. 우리는 상태를 "후보별 거짓 수의 다중집합"으로 축약한 정확 DP를 만들고, 모든 최적 전략을 정의 그대로 재시뮬레이션하는 인증서와 쌍대(예산) 재귀, 그리고 e=0 폐형식으로 삼중 검증했다. 검증 스택의 검출력을 측정하기 위해 알려진 버그 6종을 일부러 주입했고, 4종은 세 검사가 모두 잡아냈으며 1종은 부분적으로, 1종(행동 차이가 없는 변이)은 잡지 못했다. 이론에서는 e=0일 때 C(n,0,p)가 c*·log₂n ≤ C ≤ c*·log₂n + p_max (c*=min_k p_k/log₂k) 샌드위치를 만족함을 증명하고, 거짓 예산이 있을 때 단일 문항 전략의 상한 p_k·L_k(n,e)를 제시한다. 실험은 가격 6종 × n≤20 × e≤2 × 문항수 상한 2·3의 120개 정확 인스턴스, n≤32 거짓 예산 스윕 78개, 원소별 가산비용 72개, n≤256 이진 전용 11개, 가격곡률 ρ 스윕 33개, 한쪽 거짓 19개로 구성된다. 핵심 발견은 네 가지다. (i) e=0에서 정확 최적은 비트당 비용 c*=min p_k/log₂k의 예측과 일치한다(정리와 수치 모두). (ii) 거짓 예산이 커지면 비트당 비용 법칙은 완전히 무력해지고(e=1, n=20에서 하한 4.32 대 실제 7.58) 최적 문항수는 작은 쪽으로 이동한다(ρ=0.8, n=12: 평균 문항 3.5→3.0→2.9). (iii) 거짓이 한 방향으로만 가능하면 같은 보장을 약 20% 싸게 얻는다(n=64, e=1: 10→8). (iv) 가격을 무시한 "질문 수 최소화" 전략은 실제 가격에서 최대 27% 더 든다. 우리는 혼합 문항수로의 자연스러운 비용 가중 구면 한계가 거짓임을 보이는 반례(n=12, e=2에서 한계 15.523 대 인증된 최적 15.479)를 제시하고, 이 중 무엇이 고전 결과이고 무엇이 새 계산인지를 절마다 구분해 적었으며, "상전이"라는 표현과 구면 한계의 비tight성에 대한 새로움 주장을 스스로 철회했다.
%ENDABS
%ABSEN
Questions have different prices, answers sometimes lie, and the width of a question (how many answers it offers) is itself a design choice. This paper studies the resulting search problem: identify one of n items when (i) a question with arity k costs p_k, (ii) a responder may lie at most e times in total, and (iii) the objective is the worst-case total cost. The model contains Rényi–Ulam search with lies (no prices, binary answers) and group testing (fixed arity, uniform cost) as its two ends. We give an exact dynamic program whose state is the multiset of per-candidate lie counts, attach to every optimum a certificate that re-simulates the strategy from the definition, and cross-check it with a dual budget recursion and, for e=0, with a closed form. To measure the detection power of this stack we injected six known bugs: four were caught by all three checks, one partially, and one (behaviourally identical) not at all. On the theory side we prove the sandwich c*log₂n ≤ C(n,0,p) ≤ c*log₂n + p_max with c*=min_k p_k/log₂k, and give the single-arity upper bound p_k·L_k(n,e) for e≥1. Experiments comprise 120 exact instances over six price families with n≤20 and arity cap 2 or 3, 78 lie-budget instances up to n=32, 72 additive-cost instances, 11 binary-only instances up to n=256, 33 price-curvature instances and 19 one-sided-lie instances. Four findings stand out: the exact optimum at e=0 agrees with the cost-per-bit prediction; once lies are present the cost-per-bit law collapses (n=20, e=1: bound 4.32, optimum 7.58) and the optimal width shifts down (ρ=0.8, n=12: mean arity 3.5→3.0→2.9); one-sided lies buy about 20% (n=64, e=1: 10→8); and count-optimal play costs up to 27% more under real prices. We also exhibit a counterexample showing that the natural cost-weighted sphere bound fails for mixed arities (n=12, e=2: bound 15.523 against a certified optimum of 15.479); each section separates what is classical from what this computation adds, we explicitly withdraw the word "phase transition" and any claim of novelty for the non-tightness of the sphere bound.
%ENDABS

# 1. 서론

질문에는 값이 붙는다. 의사의 검사, 실험실의 풀검사, 사용자의 스위치 입력, 현미경의 시야 이동은 모두 "무엇을 물을지"와 "그 질문이 얼마인지"가 분리된 결정이다. 어떤 질문은 한 번에 많은 후보를 갈라놓지만 비싸고, 어떤 질문은 싸지만 조금밖에 가르지 못한다. 그리고 대답은 가끔 틀린다.

이 상황을 가장 단순하게 적으면 세 개의 축이 남는다. 첫째, 질문의 **문항수**(arity) k, 즉 몇 가지 답을 주는가. 둘째, 그 질문의 **가격** p_k. 셋째, 대답이 거짓일 수 있는 **예산** e. 목적은 최악의 경우 총비용을 최소화하는 것이다.

이 세 축을 동시에 다룬 문헌은 우리가 조사한 범위에서 없었다. 거짓말 탐색 문헌은 문항수를 2로 고정하고 가격을 균일하게 두며 질문 수를 센다 [P8413DD, PF15DF0, P14F31E]. 그룹테스트 문헌은 문항수 구조를 고정하고 비용을 균일하게 두거나, 두 종류 검사의 비용비만 다룬다 [P5B0ED3, PA8FD08]. 결정트리 문헌은 비용을 다루지만 잡음이 없고 대개 기대 비용을 목적함수로 둔다 [PC4925D, P1A2D80]. 본 논문은 이 셋을 한 모형에 넣고, 작은 규모에서 **정확히** 계산한다.

질문은 다음 하나다. **가격 p와 거짓 예산 e가 주어졌을 때, 최악 비용 C(n,e,p)는 얼마이며, 그 최적 전략은 어떤 폭의 질문을 던지는가?**

기여는 다음 넷이다.

- **모형.** 문항수 가격이 있는 q진 거짓말 탐색을 정의한다(정의 1–4). 이 모형은 균일 가격·이진 답에서 고전 Rényi–Ulam 게임을, e=0에서 비용 있는 결정트리를, 문항수를 n으로 고정하면 "한 번의 n지 선택"을 포함한다.
- **정확 알고리즘과 인증서.** 상태를 후보별 거짓 수의 다중집합으로 축약해 상태 공간을 조합적으로 줄인 DP(정리 2), 모든 최적 전략을 정의 그대로 재시뮬레이션하는 인증서(5.2절), 쌍대 예산 재귀와 e=0 폐형식에 의한 삼중 검증을 제시한다. 검증 스택 자체의 검출력을 버그 주입으로 측정했다(5.3절).
- **정리.** e=0에서 C(n,0,p)가 c*·log₂n과 c*·log₂n+p_max 사이에 끼인다는 것(c*=min_k p_k/log₂k)을 증명한다(정리 1). 이것이 "비트당 비용" 직관의 정확한 형태이며, 동시에 그 직관이 **어디까지** 성립하는지를 말해 준다. 거짓 예산이 있을 때는 단일 문항 전략의 상한 p_k·L_k(n,e)를 제시한다(명제 4).
- **전수 계산과 그로부터 드러난 것.** 260개가 넘는 정확 인스턴스에서 (i) e=0의 최적 문항수가 비트당 비용 곡선의 최소점과 일치하고, (ii) 거짓 예산이 커지면 그 법칙이 붕괴하며 최적 문항수가 작아지고, (iii) 한쪽 거짓은 같은 보장을 약 20% 싸게 만들고, (iv) 가격을 무시한 질문 수 최소화는 최대 27% 손해라는 것을 보인다.

논문의 태도는 보수적이다. 우리가 새로 세운 정리는 정리 1과 명제 2·4처럼 증명이 짧은 것들이고, 무게중심은 **정확 계산과 그 계산이 반박한 것**에 있다. 특히 다음 두 가지를 스스로 철회한다. 첫째, 우리는 초기 작업에서 가격곡률 ρ에 따른 "문항수 상전이"라고 불렀던 현상을, 정수성과 비트당 비용의 산술적 결과로 재규정한다(7.1절). 재검토 끝에 이 점을 받아들였다. 둘째, 구면 한계가 e=2의 작은 사례에서 tight하지 않다는 사실은 우리의 발견이 아니라 거짓말 탐색 문헌의 고전적 현상이므로, 새 결과로 주장하지 않고 재현 사실로만 보고한다(4.3절).

# 2. 관련 연구

## 2.1 거짓말 탐색: 가장 가까운 선행연구

질문자가 부분집합을 묻고 응답자가 최대 ℓ번 거짓말을 하는 게임의 최소 질문 수를 다루는 문헌이 있다. 우리 모형과 가장 겹치는 것은 **질문 크기 제한** 변형이다. 한 질문이 최대 k개 원소만 물을 수 있을 때의 최소 질문 수 RU^k_ℓ(n)이 연구되었고, "Convexity Lemma"를 통해 일반 하한과 상한이 제시되었다 [P8413DD]. 이 결과는 우리의 p_k를 모두 1로 두고 답을 이진으로 제한한 특수한 경우와 형식적으로 겹친다. 우리가 더하는 것은 **가격 p_k**와 **k진 답**, 그리고 **정확 계산**이다.

분포적 20 질문에 k개의 거짓이 섞이는 경우, 최적 전략의 평균 질문 수가 H(μ)+k·H₂(μ) 규모라는 결과가 있다 [PF15DF0]. 병리적 거짓말 게임은 relaxed Hamming ball 덮개와 동치이며 [P14F31E], 선형 비율의 거짓에 대해서는 구면 한계의 O(√(log log n)/(1−2f))배 구성이 알려져 있다 [P40E92F]. 거짓 패턴에 정규 제약을 두는 일반화 [PAB129B]와 t진 질의 + lie channel [PE7DF7A]도 있다.

**거짓의 비대칭성**을 다룬 연구로 half-lie 모형이 있다. 두 응답 중 하나만 거짓일 수 있는 상황을 현미경 표적 탐색에 적용하고, "양성 결과로 검증"을 강제하면 평균 질문이 50% 늘어난다고 보고한다 [P1DB548]. 우리의 6.6절은 이 축을 최악 비용으로 옮겨 계산한다. 마지막으로 거짓말 게임과 예측 오류를 허용하는 온라인 알고리즘을 연결한 연구가 있다 [P6C601E].

## 2.2 그룹테스트: 비용 축의 절반

그룹테스트는 d개의 결함을 풀 검사로 찾는 문제이며, 정보이론적 정리가 잘 정비되어 있다 [PA8FD08]. 적응적 그룹테스트의 용량 [P06A777], Bernoulli 비적응의 sharp 임계값과 "밀집 영역에서 적응이 엄격히 우세" [P9AD26E], 잡음이 있을 때의 근사 재구성 한계 [PE5CF06], 비대칭 잡음을 포함한 준최적 알고리즘 [P5E6A4C], 비적응 위상전이 [P08CCDD], 경쟁비 c<1.431 [P685226]과 c>1.31 하한 [PFB9004] 등이 있다.

비용을 직접 다룬 문헌도 있다. 양성 검사가 음성보다 훨씬 비쌀 수 있는 상황에서 양성:음성 비용비를 조절해 결정론적·확률적 접근을 비교한 연구가 우리의 "가격" 축과 가장 가깝다 [P5B0ED3]. 그러나 이 연구도 **문항수 자체를 가격화하지는 않는다** — 검사는 언제나 예/아니오다. 부호 쪽에서는 disjunct 행렬의 최소 행 수 [P320853], superimposed 부호의 rate [PD8BD7C], 부분집합 질의의 2단계 하한 [PCFF773]이 있다.

## 2.3 결정트리와 질문 비용

변수 읽기에 비용이 있을 때 함수를 결정하는 전략 트리를 최소 비용으로 설계하는 문제는 오래되었다. 최악 비용과 기대 비용을 **동시에** 근사하는 연구 [PC4925D], 질문 비용이 지수적으로 증가하는 경우의 GBS 일반화 [P1A2D80], 분기 비용이 결과에 따라 달라지는 조건부 비용 트리 [PB950B4], 가중 트리 탐색의 준다항 근사 [P95964D], 정수계획법 기반 최적 결정트리 [P1FCE5B], 균일 결정트리에 대한 greedy의 준지수 근사 [PE18BCE], DP 분리가능성의 필요충분조건 [P2914C1]이 있다. 응용으로 비용민감 분류 [P4A3914]와 AAC 스위치 스캐닝의 기대 질문 최소화 [P6D10EE]가 있다.

이 문헌과 우리의 차이는 두 가지다. 첫째, 대부분 **기대** 비용을 최소화하고 우리는 **최악** 비용을 최소화한다. 둘째, 이 문헌에는 **거짓말(적대적 오류)이 없다**. 잡음이 있는 결정트리 연구 [PEAF8CB]도 잡음을 확률적으로 다루며 최악 거짓 예산을 두지 않는다.

## 2.4 동전저울

받침저울 문제는 같은 질의 구조(3진 답)를 공유하는 퍼즐 문헌이다. 모든 변형의 비적응 최적해 [P396637], 세 가지 무게 동전 정렬의 ⌈3n/2⌉−2 [P34C7AC], 카멜레온 동전 [P3DC0F4], 저울 퍼즐과 그룹테스트의 통합적 시각 [PB98308]이 있다. 이 문헌은 "질문 수"를 세고 거짓을 허용하지 않는다.

## 2.5 경계

!table:표 1. 선행 연구와 본 논문의 경계. 여기서 "가격"은 문항수 의존 가격 p_k를 뜻한다.
| 항목 | [P8413DD] | [P5B0ED3] | [PC4925D] | [PA8FD08] | 본 논문 |
| 모형 | 이진, 크기 제한 | 이진, 두 검사 비용 | 결정트리 | 그룹테스트 | k진, 문항수 가격 |
| 거짓 | 최대 ℓ | 없음 | 없음 | 확률·적대 | 최대 e (전역) |
| 목적 | 질문 수 | 총비용 | 최악+기대 | 검사 수·오류 | 최악 총비용 |
| 결과 | 하한·상한 | 실험 비교 | 근사 | 점근 정리 | 정확 계산+인증서 |
| 규모 | 이론 | 중간 | 이론 | 점근 | n≤256 (이진), n≤32 (거짓) |
!endtable

# 3. 모형

**표기.** U={1,…,n}은 후보 집합, x∈U는 숨은 정답, e≥0은 거짓 예산, p=(p_2,p_3,…)는 가격 벡터(p_k>0)다.

**정의 1 (질문).** 질문 Q는 U의 **순서 있는 분할** (B_1,…,B_k)이다. 빈 블록을 허용하며, k를 Q의 **문항수**라 하고 가격은 p_k다. 정직한 답은 x가 속한 블록의 번호다.

**정의 2 (거짓 응답자).** 응답자는 전체 게임에서 최대 e번, 정직한 번호 대신 임의의 다른 번호를 답할 수 있다. 응답의 일관성은 요구된다: 어떤 답도 "어떤 후보 x와 e 이하의 거짓"으로 설명 가능해야 한다. 따라서 모든 후보를 동시에 탈락시키는 답은 일어나지 않는다.

**정의 3 (전략과 비용).** 전략은 답 이력에 따라 다음 질문을 고르는 적응적 트리다. 전략의 비용은 응답자가 실현 가능한 답을 고를 때의 총가격의 최댓값이다. 전략이 **정확**하다는 것은, 임의의 x와 e 이하의 거짓 패턴에 대해 전략이 멈추고 x를 출력한다는 뜻이다.

**정의 4 (최적 비용).** C(n,e,p) = 정확한 전략의 최소 비용(존재하지 않으면 ∞). 우리는 문항수 상한 K를 명시할 때 C_K(n,e,p)로 쓴다.

**정의 5 (상태).** 답 이력 뒤에 후보 y에 대해 d(y)를 "y를 포함하지 않는 블록으로 답한 횟수"라 하면, y가 일관적인 것은 d(y)≤e와 동치이며 d(y)는 y가 진실일 때 필요한 최소 거짓 수다. 따라서 상태는 벡터 d이고, d(y)>e인 후보는 탈락한다.

**명제 1 (상태의 충분성).** 두 이력이 같은 d 벡터를 주면, 남은 게임의 최적 비용은 같다.

*증명.* 남은 게임에서 후보 y에 대한 정보는 "지금까지 y가 거짓을 e−d(y)번 더 견딜 수 있다"는 것뿐이고, 미래의 답은 질문과 x에 의해서만 결정되며 과거 이력과 무관하다. 따라서 (d, e)가 같으면 미래의 가능한 답 집합과 비용이 같다. □

**정의 6 (대칭 가격).** p가 문항수에만 의존하면 두 후보를 바꾸는 순열이 문제를 보존하므로, 상태는 d 값의 **다중집합**, 즉 (a_0,…,a_e)(a_i=#{y:d(y)=i})로 축약된다.

# 4. 이론

## 4.1 e=0: 비트당 비용의 정확한 형태

**보조정리 1 (Kraft형).** 각 내부 노드 v의 문항수가 k(v)이고 실제로 후보가 배정된 자식 수가 d(v)≤k(v)인 결정트리에서, 잎의 수 L은 max_{잎 ℓ} Π_{v∈경로(ℓ)} d(v) 이하이다.

*증명.* 노드 v의 서브트리 잎 수를 N(v)라 하면 N(v)=Σ_{자식 c} N(c) ≤ d(v)·max_c N(c)이고, 귀납적으로 N(v) ≤ Π_{u∈경로} d(u)이다. 루트에서 N(루트)=L. □

**정리 1 (e=0의 폐형식).** 2≤k_j≤K일 때
$$C_K(n,0,p)  =  min{ Σ_j p(k_j) : Π_j k_j ≥ n }$$
이고, c* = min_{2≤k≤K} p_k/log₂k라 하면 따름정리로
$$c*·log2 n  ≤  C_K(n,0,p)  ≤  c*·log2 n + max(k≤K) p_k$$
를 얻는다.

*증명 (하한).* 임의의 정확한 전략 T를 보자. 후보가 다르면 답 수열도 달라야 하므로 잎은 n개다. 보조정리 1에서 n ≤ max_경로 Π k_j이므로, 곱이 최대인 경로 π는 Π_{j∈π} k_j ≥ n을 만족한다. 그 경로의 수열은 정리의 최소화 문제에서 **실현 가능한 후보**이고, 그 경로의 비용은 worst-case 비용 이하다. 따라서 min{Σ p : Π k ≥ n} ≤ (경로 π의 비용) ≤ cost(T)이고, T에 대해 최소를 취하면 우변의 하한을 얻는다. 이어서 c*의 정의로
$$Σ(j∈π) p(k_j) = Σ(j∈π) [p(k_j)/log2 k_j]·log2 k_j  ≥  c*·Σ(j∈π) log2 k_j  =  c*·log2(Π(j∈π) k_j)  ≥  c*·log2 n$$
*증명 (상한).* c*=p_{k*}/log₂k*를 달성하는 k*(≤K)를 택하고 L=⌈log_{k*} n⌉로 두면 (k*)^L ≥ n이다. (k*)^L 격자의 서로 다른 n개 칸에 후보를 하나씩 주입하고, j번째 질문을 "j번째 좌표가 같은 칸끼리 묶는" 분할로 정의하면 모든 경로가 같은 수열 (k*,…,k*)을 가지며 비용은
$$L·p_{k*}  ≤  p_{k*}·(log(k*) n + 1)  =  c*·log2 n + p_{k*}  ≤  c*·log2 n + max(k≤K) p_k$$
이다. □

정리 1은 두 가지를 말한다. 첫째, e=0에서 **최적 문항수는 c*=p_k/log₂k를 최소화하는 k** 근처이며, 정수성 때문에 계단이 생긴다. 둘째, 하한은 근사적이지 않고 **비율 1**로 맞는다(오차는 상수항 p_max뿐). 우리는 6.4절에서 이 예측을 정확 DP와 비교한다.

## 4.2 단조성

**명제 2 (단조성).** C(n,e,p)는 e와 n에 대해 비감소이고, 가격이 점별로 커지면(p'_k≥p_k ∀k) 비감소한다.

*증명.* e가 늘면 거짓 예산 e+1에 대한 전략은 e에 대해서도 정확하므로 최소값이 줄지 않는다. n이 늘면 U에 원소를 하나 추가한 인스턴스이고, 추가 원소를 구별해야 하므로 줄지 않는다. 가격이 커지면 같은 전략의 비용이 커지므로 최소값도 커진다. □

주의: 가격 단조성은 **점별** 순서에서만 성립한다. 평균 가격이 커지는 것만으로는 성립하지 않는다(예: p_2만 커지고 p_3이 작아지면 최적이 3지로 옮겨가 총비용이 줄 수 있다).

## 4.3 구면 한계: 고전 결과의 재현

**명제 3 (구면 한계, 고전).** 모든 질문의 문항수가 k이고 깊이가 L 이하인 전략이 e개의 거짓을 견디려면
$$n·Σ(i=0..e) C(L,i)·(k−1)^i  ≤  k^L$$

*증명.* 후보 x의 정직한 답 수열에서 i개 위치를 거짓으로 바꾸면 그 위치마다 k−1가지 선택이 있으므로, x와 e 이하로 일치하는 수열이 Σ_i C(L,i)(k−1)^i개다. 두 후보의 이런 수열 집합이 겹치면 그 수열은 두 후보 모두와 e 이하로 일치해 전략이 틀린다. 전체 수열은 k^L개이므로 합이 k^L 이하. □

이 한계는 **필요조건일 뿐**이며 tight하지 않다. 우리 계산에서 n=4, e=2, k=2일 때 한계는 L=7을 허용하지만(4·(1+7+21)=116≤128) 정확 최적은 8이다. 이 현상은 거짓말 탐색 문헌에서 오래 알려진 것이며 [P8413DD 등의 배경], **우리의 발견이 아니다.** 우리는 이를 새 결과로 주장하지 않고, 구현이 고전적 비tight성을 재현한다는 검증 사실로만 쓴다.

## 4.4 거짓 예산이 있을 때: 단일 문항 상한

**명제 4 (단일 문항 상한).** k를 고정하고 L_k(n,e)=min{L : nΣ_{i≤e}C(L,i)(k−1)^i ≤ k^L}이라 하면
$$C(n,e,p)  ≤  min(k) p_k·L_k(n,e)$$

*증명.* 명제 3의 부등식을 만족하는 L을 택하고, k진 균형 전략(각 단계에서 후보를 k개 블록에 가능한 균등하게 나눔)을 쓰면 Σ_i C(L,i)(k−1)^i ≥ n을 만족하므로 e개 거짓을 견딘다. 문항수는 L이고 비용은 p_k L. 최소를 취한다. □

명제 4는 상한이므로, 정확 최적과의 차이는 "혼합 문항수와 비균형 분할이 주는 이득"이다. 6.4절에서 이 차이를 측정한다.

## 4.5 혼합 문항수로의 자연스러운 확장은 거짓이다

명제 3을 혼합 문항수로 확장하는 가장 자연스러운 추측은 다음과 같다.

**추측 (비용 가중 구면 한계).** 임의의 e와 가격 p에 대해
$$C(n,e,p) ≥ min{ Σ_j p(k_j) : n·Σ(i≤e) E_i(k_1..k_L) ≤ Π_j k_j }$$
여기서 E_i는 (k_j−1)들의 i차 기본대칭다항식이다.

이 추측은 **거짓이다.** n=12, e=2, K=4, p=(1.741, 2.408, 3.031)(즉 p_k=k^{0.8})에서 우변의 값은 15.523인데, 인증서로 검증된 정확 최적은 15.479다. 반례의 원인은 구조적이다. 거짓 답은 전략을 **다른 경로**로 보내므로, 후보 x의 "거짓 공"(반경 e의 답 수열 집합)은 x 자신의 경로 문항수만으로 세어지지 않는다. 균일 문항수에서는 모든 경로의 문항수가 같아 이 문제가 사라지지만, 혼합에서는 사라지지 않는다. 우리는 231개 정확 인스턴스에서 이 추측을 검사해 위반 1건을 찾았고, 그 1건을 위에 적었다.

이 반례는 논문의 주장을 약화시키는 것이 아니라 **어디가 어려운지**를 정확히 말해 준다. e≥1에서 비용 가중 하한을 만들려면 경로 의존성을 다루는 새 도구가 필요하며, 우리는 그것을 갖고 있지 않다(8.1절).

# 5. 알고리즘과 검증

## 5.1 대칭 상태 DP

**정리 2 (정확성과 복잡도).** 정의 6의 상태 (a_0,…,a_e) 위에서
$$V(a) = min(2≤k≤K) min(m) [ p_k + max(b : m(·,b)≠0) V(Φ(a,m,b)) ],   단 Φ(a,m,b)=a 이면 ∞$$
(단, Φ(a,m,b)=a이면 그 가지는 ∞) 를 계산하면 V((n,0,…,0))=C_K(n,e,p)다. 여기서 m은 각 계급의 후보를 k개 블록에 배분하는 행렬이고, Φ는 "블록 b에 든 후보는 계급 유지, 나머지는 계급+1(계급 e 초과는 탈락)"이다. 상태 수는 n을 e+2개 부분으로 나누는 약한 합성의 수 O(n^{e+1})이고, 전이는 블록 순열을 접어 세면 상태당 유한하다. 따라서 e를 고정하면 n에 대해 다항시간(XP)이다.

*증명.* 명제 1에서 상태가 충분통계량이므로 벨만 방정식이 성립한다. 응답자는 실현 가능한 답(블록에 후보가 있는 답) 중 최악을 고르고, 답이 상태를 바꾸지 않으면 응답자가 영원히 같은 답을 되풀이할 수 있으므로 그 질문은 무효다. 정지 조건은 "일관적 후보가 정확히 하나"이며, 이때 그 후보가 정답이다. □

구현에서 블록 순열을 접는 것이 결정적이다. 문항수 k의 배분 행렬은 원래 k!개의 순열을 갖지만 응답자는 블록에 대한 최댓값만 취하므로 순서는 의미가 없다. 접지 않으면 n=20, e=1, K=5에서 한 상태당 수백만 개를 세게 되고, 접으면 같은 계산이 수 초 안에 끝난다(6.9절).

## 5.2 인증서와 삼중 검증

DP는 최적 전략을 "상태 → (문항수, 배분행렬)" 사전으로 돌려준다. 우리는 세 가지 독립 검사를 쓴다.

- **D1 인증서 재시뮬레이션.** 전략의 도달 가능한 상태 그래프를 걸으며 (i) 종료 상태가 정확히 하나의 일관적 후보를 갖는지, (ii) 각 상태의 질문이 실현 가능한 모든 답을 처리하는지, (iii) 어떤 답도 상태를 정체시키지 않는지 확인하고, 그래프 위에서 최악 비용을 다시 계산해 DP 값과 비교한다. 추가로 작은 n에서는 상태를 실제 원소로 구체화해 정의 그대로 시뮬레이션한다(concrete simulator).
- **D2 쌍대(예산) 재귀.** "비용 ≤ B로 보장 가능한가"를 묻는 별도 재귀를 돌려, B=C−ε에서는 불가능, B=C에서는 가능함을 확인한다. 이는 최소화 대신 예산을 내려보내는 다른 알고리즘이므로 비용 계산 실수를 잡는다.
- **D3 e=0 폐형식.** 정리 1의 최단경로 계산(곱 ≥ n인 수열의 최소 비용)과 비교한다.

실행 결과: A단계 120개, B단계 78개 인스턴스 전부에서 D1이 통과했고(불일치 0), 무작위 25개에서 D1과 D2가 일치했으며, 무작위 30개에서 e=0의 DP와 D3가 오차 0으로 일치했다. 또한 고전 검증으로 q(2,e)=2e+1(e≤3)과, 이진·균일 가격에서 n≤16, e≤2의 값이 명제 3의 한계와 일치함을 확인했다(단, 4.3절의 비tight 사례 제외).

## 5.3 검증 스택의 검출력 감사

검사가 있다는 것과 검사가 **작동한다**는 것은 다르다. 우리는 알려진 버그 6종을 DP에 주입하고 세 검사가 실제로 잡는지 측정했다(6개 인스턴스 × 7변형).

!table:표 2. 버그 주입 감사. "검출"은 검사가 주장값과 불일치를 보고한 경우다.
| 변이 | 내용 | D1 | D2 | D3 |
| terminal2 | 후보 2개를 종료로 오인 | 6/6 | 6/6 | 2/2 |
| cost_once | 비용을 루트에서 한 번만 합산 | 6/6 | 6/6 | 2/2 |
| min_branch | 응답자의 최댓값을 최솟값으로 | 6/6 | 6/6 | 2/2 |
| wrong_side | 거짓을 블록 안쪽 후보에 계산 | 6/6 | 6/6 | 2/2 |
| no_cap | 계급 상한(탈락)을 무시 | 4/6 | 4/6 | 0/2 |
| no_stall | 정체 답을 허용 | 0/6 | 0/6 | 0/2 |
| (정상) | 주입 없음 | 0/6 | 0/6 | 0/2 |
!endtable

읽는 법: 네 개의 변이는 세 검사가 모두 즉시 잡는다. `no_cap`은 일부 인스턴스에서 우연히 같은 값을 내 4/6이다. `no_stall`은 어느 검사도 잡지 못했는데, 이 변이는 정체 답을 허용해도 이 인스턴스들에서 최적값이 달라지지 않았기 때문이다(행동 차이가 없는 변이). 이 음성 결과는 검증 스택의 한계를 정직하게 말해 준다: **검사는 관측 가능한 오류를 잡지만, 최적값을 바꾸지 않는 구현 차이는 잡지 못한다.**

# 6. 실험

## 6.1 설정

가격 가족 6종을 쓴다: p_k=1(단위), log₂k(정보 비례), √k, k(선형), 1+log₂k(고정+정보), 그리고 "이진 1, 그 외 2"(step). n∈{8,12,16,20}, e∈{0,1,2}, 문항수 상한 K∈{2,3}로 120개 정확 인스턴스를 계산했다. 별도로 거짓 예산 스윕(n≤32, e≤3, 78개), 원소별 가산비용(그룹테스트형, 72개), 이진 전용 대규모(n≤256, 11개), 가격곡률 ρ∈[0,2](n=12, K=4, 33개), 한쪽 거짓(19개)을 계산했다. 모든 인스턴스는 인증서 검사를 통과했고, 실패·시간초과·오류는 결과 파일에 그대로 남겼다.

## 6.2 문항수 풍경: 문항수를 늘리면 얼마나 싸지는가

!fig:figures/fig2_value_of_arity.png|그림 1. 문항수를 3까지 허용할 때 이진 전용 대비 절감률(n=20). 가격이 문항수에 대해 볼록할수록(√k, k) 절감이 크고, step 가격에서는 0이다.

가격이 문항수에 대해 선형 이하로 증가하면 3지 질문이 이득이다. n=20, e=0, p_k=√k: 7.07→5.20(26%), p_k=k: 10.00→9.00(10%). 반면 step 가격(이진 1, 그 외 2)에서는 모든 e에서 절감이 정확히 0이다 — 이진이 이미 단위 비용당 최선이기 때문이며, 이는 정리 1의 예측(c*가 k=2에서 달성됨)과 일치한다. p_k=1(단위)에서는 이득이 가장 크다(11.00→7.00, 36%). 단위 가격에서 문항수를 늘리는 것은 "같은 값에 더 많은 정보"이므로 당연하지만, 그 크기가 e에 따라 커진다는 점은 주목할 만하다(같은 n=20에서 e=0 40%, e=1 37.5%, e=2 36% — 대체로 비슷하다).

## 6.3 거짓 예산: 비용은 어떻게 자라는가

!fig:figures/fig4_lie_sweep.png|그림 2. 거짓 예산 e에 대한 최적 비용. 실선은 문항수 상한 3, 점선은 2.

이진 전용·단위 가격에서 n=128, e=1의 최적은 11로, 명제 3의 한계와 정확히 일치한다(128·(1+11)=1536≤2048, 128·11=1408>1024). e=0에서는 log₂n=7이므로 **거짓 하나가 4개의 질문을 더 요구한다**. n=16에서는 4→7(+3), n=64에서는 6→10(+4)로, 추가 비용이 log₂log₂n 규모로 아주 천천히 자란다. 이는 고전 결과와 정합적이며 우리의 기여는 그것을 비용 축에서 재확인한 것이다.

!fig:figures/fig7_growth.png|그림 3. n에 대한 이진 질문 수(실측). 점선은 구면 한계로, 대부분의 점에서 실측과 일치한다.

## 6.4 비트당 비용 법칙은 어디까지 성립하는가

정리 1은 e=0에서 c*=min p_k/log₂k가 비용을 지배한다고 말한다. 표 3은 정확 DP와 c*·log₂n을 비교한다.

!table:표 3. e=0에서 정확 최적과 비트당 비용 하한. p_k=1+log₂k (K=3).
| n | 정확 최적 | c*·log₂n | 상한 c*·log₂n+p_max |
| 8 | 3.00 | 3.00 | 4.00 |
| 12 | 3.58 | 3.58 | 4.58 |
| 16 | 4.00 | 4.00 | 5.00 |
| 20 | 4.58 | 4.32 | 5.32 |
!endtable

네 경우 모두 하한과 상한 사이에 있고, 두 경우는 하한에 정확히 닿는다. n=20에서 0.26의 간격이 남는데, 이는 정수성(문항수는 정수, 가격은 실수) 때문이며 정리 1의 상한 p_max가 이를 덮는다.

반면 **거짓이 들어오면 이 법칙은 붕괴한다.** 같은 가격 가족에서 e=1, n=20의 정확 최적은 7.58인데 c*·log₂n=4.32다. 즉 비트당 비용은 하한조차 아니게 된다(정리 1은 e=0 전용이다). 이 붕괴의 크기는 e와 함께 커지며, 실용적 의미는 분명하다: **잡음이 있는 세계에서는 "정보당 가격"으로 질문을 고르는 것이 위험하다.**

!fig:figures/fig5_bound_vs_exact.png|그림 4. 균일 문항 구면 한계와 정확 최적의 산점도. 점이 대각선 위에 있으면 한계가 비tight하다는 뜻이다.

## 6.5 가격곡률과 최적 문항수

!fig:figures/fig1_phase.png|그림 5. (a) 가격곡률 ρ에 대한 최적 비용, (b) 최악 경로의 평균 문항수(n=12, K=4).

p_k=k^ρ에서 ρ를 0에서 2로 올리면 최적 문항수가 3~4에서 2로 내려간다. 초기 작업에서 우리는 이를 "상전이"라 불렀으나, 자체 재검토를 거쳐 철회한다. 정리 1이 이미 말하듯 최적 문항수는 c(ρ)=min_k k^ρ/log₂k의 최소점 근처이고, 그 최소점은 k*=e^{1/ρ}다. ρ=0.6이면 k*≈5.3, ρ=0.8이면 3.5, ρ=1.0이면 2.72, ρ=1.2면 2.3이다. 관측된 문항수는 이 값들을 정수로 자른 것과 일치한다. 따라서 이것은 **상전이가 아니라 비트당 비용 곡선과 정수성의 산술**이다. 다만 다음 두 가지는 산술만으로 예측되지 않는다.

- **거짓 예산의 하향 이동.** ρ=0.8, n=12에서 평균 문항수는 e=0에서 3.5, e=1에서 3.0, e=2에서 2.86이다. 거짓이 늘면 (k−1)^e 항이 커지므로 넓은 질문의 이점이 줄어든다. 이 이동은 정확 계산으로만 정량화된다.
- **혼합의 이득은 작다.** ρ=0.8, e=2에서 최악 경로의 문항수는 [2,2,3,2,4,3,3]으로 섞여 있지만, 단일 문항 전략(명제 4)과의 차이는 표 4에서 0.3~0.6에 불과하다.

!table:표 4. 단일 문항 상한(명제 4)과 정확 최적의 차이. p_k=k^ρ, n=12, K=4. 값은 "상한 − 정확 최적".
| ρ | e=0 | e=1 | e=2 |
| 0.0 | 0.00 | 0.00 | 0.00 |
| 0.2 | 0.07 | 0.15 | 0.32 |
| 0.4 | 0.19 | 0.38 | 0.80 |
| 0.6 | 0.36 | 0.73 | 1.26 |
| 0.8 | 0.62 | 1.33 | 1.38 |
| 1.0 | 1.00 | 1.00 | 1.00 |
| 1.2 | 0.86 | 0.86 | 0.86 |
| 1.4 | 0.62 | 0.62 | 0.62 |
| 1.6 | 0.26 | 0.26 | 0.26 |
| 1.8 | 0.00 | 0.00 | 0.00 |
| 2.0 | 0.00 | 0.00 | 0.00 |
!endtable

읽는 법: 차이가 0인 양 끝(ρ=0, ρ≥1.8)에서는 단일 문항 전략이 이미 최적이다. 반면 중간 곡률 ρ=0.8, e=2에서 차이는 1.38로 최적(15.48)의 약 9%다. 즉 **혼합은 곡률이 중간일 때만 의미가 있고**, 그 크기는 무시할 수 없다. 우리의 초기 가설("혼합이 전 구간에서 중요하다")과 반대 가설("혼합은 언제나 무의미하다")은 **둘 다 틀렸다** — 정확 계산이 그 경계를 그어 준다.

## 6.6 원소별 가산비용: 싼 질문이 나쁜 질문일 때

그룹테스트형 모형에서 질문 "x∈A인가"의 가격을 A의 원소 가중치 합으로 두면, 문항수는 항상 2지만 질문마다 가격이 다르다. n=5,6,7에서 무작위 가중치(로그정규)와 두 단계 가중치를 72개 인스턴스로 계산했다.

!fig:figures/fig6_additive.png|그림 6. 가장 싼 정보성 질문의 가격(가로)과 최적이 실제로 먼저 던지는 질문의 가격(세로). 대각선 아래에 있는 점은 "최적이 더 싼 질문을 택했다"는 뜻이고, 위는 그 반대다.

관찰: 최적은 거의 항상 가장 싼 질문을 고르지 않는다. "한 개의 저렴한 원소" 단면(가중치 0.1 하나 + 10이 여섯)에서 e=0의 최적 비용은 50.1인데, 이는 10짜리 원소를 다섯 번 나누는 값이다. 0.1짜리 원소를 먼저 물으면 그 원소 하나만 분리되고 나머지 여섯을 여전히 갈라야 하므로, 싼 질문은 **정보가 거의 없다**. 이는 그룹테스트 문헌의 "비용 인식" 문제 [P5B0ED3]와 같은 현상이지만, 여기서는 문항수가 아니라 원소 가중치가 원인이며, 최악 비용에서 정확히 계산되었다.

## 6.7 한쪽 거짓: 비대칭 잡음의 값

지금까지는 어떤 답이든 거짓일 수 있었다. 실제로는 "양성이라고 말했는데 사실 음성"이 불가능한 경우가 많다(검사가 놓칠 수는 있어도 없는 것을 만들지는 않는다). 이 모형에서 "예"는 항상 진실이므로, 응답자는 침묵(아니오)으로 숨는 쪽을 택한다.

!table:표 5. 같은 보장을 얻는 최소 이진 질문 수: 양방향 거짓 대 한쪽 거짓.
| n | e=1 양방향 | e=1 한쪽 | e=2 양방향 | e=2 한쪽 |
| 8 | 6 | 5 | 9 | 6 |
| 16 | 7 | 6 | 10 | 8 |
| 32 | 9 | 7 | 11 | 9 |
| 64 | 10 | 8 | — | 10 |
!endtable

한쪽 거짓은 n=64, e=1에서 10 대신 8로 약 **20%** 싸고, e=2에서는 격차가 더 크다(16에서 10 대 8). half-lie 문헌이 평균 질문에서 관찰한 비대칭성 [P1DB548]이 최악 비용에서도 유지되며, 그 크기가 대략 2개의 질문임을 정확 계산으로 확인했다. (표의 "—"는 시간 내 계산하지 못한 칸이다.)

## 6.8 절약의 대가

가격을 무시하고 질문 수만 최소화하는 전략은 얼마나 손해인가.

!fig:figures/fig3_price_of_thrift.png|그림 7. 질문 수 최소화 전략을 실제 가격으로 평가했을 때의 초과 비용(n=20).

step 가격(이진 1, 그 외 2)에서 질문 수 최소화는 3지 질문을 쓰고, 그 결과 e=2, n=20에서 14 대 최적 11로 **27%** 더 든다. 일반적으로 이 손해는 가격에 제약이 없으면 **유계가 아니다**(p_2=1, p_3=M이면 손해는 M 규모). 따라서 "27%"는 우리가 조사한 가격 가족 안에서의 값이며, 정리로 주장하지 않는다.

## 6.9 계산 비용

!fig:figures/fig8_runtime.png|그림 8. e=2에서 n에 대한 계산 시간(로그 눈금).

블록 순열 접기 전에는 n=20, e=1, K=3이 100초를 넘었고, 접은 뒤에는 같은 인스턴스가 0.6초였다. n=40, e=1, K=3은 24초, n=20, e=1, K=5는 97초, n=128, e=1(이진)은 36초였다. n≥40에서 e≥2, K≥3인 조합은 시간 내 계산하지 못했고 결과 파일에 남기지 않았다(계산하지 않은 값을 적지 않는다).

# 7. 논의: 확인된 것, 반박된 것, 실패한 것

## 7.1 우리가 철회한 주장

- **"문항수 상전이".** 6.5절에서 보인 바와 같이, 가격곡률에 따른 문항수 변화는 비트당 비용 곡선과 정수성으로 설명된다. "상전이"는 과장이므로 쓰지 않는다. 남는 것은 (i) 정확 계산이 그 산술을 확인한다는 것, (ii) 거짓 예산이 문항수를 하향 이동시킨다는 것, (iii) 혼합의 이득이 작다는 것(우리 가설의 반박)이다.
- **구면 한계 비tight성의 새로움.** 4.3절의 n=4, e=2 사례는 고전적 현상이다. 우리는 이를 "관찰"이 아니라 "구현 검증"으로만 보고한다.
- **"27% 손해"의 일반성.** 가격 제약이 없으면 임의로 커질 수 있으므로, 특정 가격 가족에서의 측정값으로 한정한다.

## 7.2 확인된 것

- e=0에서 C(n,0,p)는 정확히 min{Σp : Πk≥n}이고, 그 따름정리로 c*·log₂n ≤ C ≤ c*·log₂n+p_max가 성립한다(정리 1, 30개 무작위 인스턴스에서 폐형식과 오차 0).
- 거짓 예산은 비용을 대략 (2e+1) 규모로 밀어올리며, 이진·단위 가격에서는 구면 한계가 소규모에서 tight하다(6.3절).
- 한쪽 거짓은 최악 비용에서도 약 20% 이득이다(6.7절).
- 문항수를 늘리는 이득은 가격의 볼록성에 의해 결정되며, step 가격에서는 정확히 0이다(6.2절).

## 7.3 실패한 시도

- **"혼합은 언제나 무의미하다"는 반대 가설.** 처음 계산한 가격 가족(√k, k, step)에서는 단일 문항 상한과의 차이가 0.6을 넘지 않아 혼합을 무시할 뻔했다. 그러나 곡률을 연속적으로 훑자(표 4) ρ=0.8, e=2에서 차이가 1.38까지 벌어졌다. 가격 가족 몇 개만 보고 일반화한 것이 잘못이었고, 곡률 스윕이 이를 잡아냈다.
- **n≥40, e≥2, K≥3.** 상태 수 O(n^{e+1})과 전이 열거 때문에 시간 내 불가능했다. 점근 거동을 이 규모에서 관측하지 못했다.
- **비용 가중 구면 한계 추측.** 명제 3을 혼합 문항수로 확장한 추측을 세우고 231개 인스턴스에서 검사했으나 1건에서 반례가 나왔다(4.5절). 추측을 철회하고 반례로 보고한다.
- **대규모에서의 최적값.** n≥256, e≥2는 이진 전용으로도 계산하지 못했다. 그림 3의 오른쪽 끝은 e≤1이다.

# 8. 한계

1. **e≥1의 정리가 없다.** e≥1에 대해 우리가 증명한 것은 상한(명제 4)뿐이고, 하한은 균일 문항에 대한 고전 구면 한계(명제 3)뿐이다. 혼합 문항수 하한은 4.5절의 반례가 보여 주듯 경로 의존성 때문에 자명하지 않으며, 우리는 이를 열린 문제로 남긴다. 따라서 이 논문은 이론 논문이 아니라 **정확 계산과 그 계산이 드러낸 현상에 관한 논문**으로 읽혀야 한다.
2. **가격 구조의 제약.** p_k가 임의일 때 성립하는 것은 명제 2(점별 단조성)뿐이다. 볼록·오목 가격에서 최적 분할의 구조적 성질은 증명하지 못했다.
3. **목적함수.** 우리는 최악 비용만 다룬다. 기대 비용과의 trade-off [PC4925D]는 계산하지 않았다. 기대 비용은 사전분포가 필요하고, 거짓 예산과 결합하면 목적함수의 정의 자체가 미묘해진다.
4. **문헌.** 이 실행 환경에서 Crossref·OpenAlex·Semantic Scholar가 차단되어 pre-arXiv 고전(Rényi 1961, Berlekamp 1968, Pelc의 survey 등)의 서지를 직접 확인하지 못했다. 본 논문의 인용은 모두 arXiv 코퍼스(2,630편) 안에서 확인한 문헌으로 제한했고, 고전은 "이름"으로만 언급하거나 2차 문헌을 통해 인용했다. 서지 세부를 추정해 적지 않았다.
5. **문헌 조사의 깊이.** 2,630편을 수집·분류했으나, 개별 초록을 정독한 것은 110편(심층 48, 요지 62)이고 나머지는 제목·서지 수준이다. 따라서 "이 주제가 완전히 새롭다"는 주장은 하지 않으며, "우리가 조사한 범위에서 이 세 축을 동시에 정확히 계산한 사례가 없다"로 한정한다.
6. **검증의 한계.** 5.3절의 `no_stall` 변이는 잡히지 않았다. 검사는 관측 가능한 오류만 잡는다.

# 9. 결론

질문의 가격, 문항수, 거짓 예산이 얽힌 탐색에서 우리는 상태를 거짓 수의 다중집합으로 축약해 정확히 계산했고, 그 계산이 무엇을 말하는지 정리와 수치로 나누어 적었다. 가장 단단한 결과는 e=0의 샌드위치(정리 1)와, 거짓이 들어오면 비트당 비용 법칙이 붕괴한다는 정량적 관찰이다. 가장 정직한 결과는 우리가 처음에 내세웠던 "상전이"와 "혼합의 중요성"이 각각 산술과 반증으로 대체되었다는 것이다.

남은 질문은 분명하다. 거짓 예산이 있을 때 임의의 가격 벡터에 대한 점근 최적 비용은 무엇인가. 혼합 문항수 전략이 점근적으로 단일 문항보다 엄격히 나은 경우가 있는가. 그리고 거짓이 시간에 따라 변하는(처음에는 정직하고 나중에 거짓인) 모형에서 최적 전략은 어떤 모양인가. 이 셋은 우리가 이번에 계산하지 못한 것이며, 다음 계산의 대상이다.

# 참고문헌

아래 문헌은 모두 본 연구의 코퍼스(`research_paper_2/lit/papers.jsonl`, 2,630편)에 존재하며, 인용은 초록에서 확인한 범위로 제한했다.

[P8413DD] Fraknói, Márton, Simon. On the Rényi-Ulam Game with Restricted Size Queries. Discrete Optimization 48:1 (2023) 100772. arXiv:2104.01664.
[PF15DF0] Dagan, Filmus, Kane. The entropy of lies: playing twenty questions with a liar. arXiv:1811.02177.
[P4DDF89] Filmus, Mehalel. Optimal sets of questions for Twenty Questions. arXiv:2106.01737.
[P14F31E] Ellis, Ponomarenko, Yan. The Renyi-Ulam pathological liar game with a fixed number of lies. arXiv:0407504.
[P40E92F] Cooper, Ellis. Linearly Bounded Liars, Adaptive Covering Codes, and Deterministic Random Walks. arXiv:0909.0029.
[PE7DF7A] Ellis, Nyman. Two-batch liar games on a general bounded channel. arXiv:0903.4866.
[PAB129B] Beluhov. Renyi-Ulam Games and Forbidden Substrings. arXiv:1609.07367.
[P1DB548] Drumm, Greentree. Microscopy as a statistical, Rényi-Ulam, half-lie game. Scientific Reports 7:14652 (2017). arXiv:1704.05980.
[P25C636] Peczarski. Strategy in Ulam's Game and Tree Code Give Error-Resistant Protocols. arXiv:0410043.
[P6C601E] Angelopoulos, Kamali. Rényi-Ulam Games and Online Computation with Imperfect Advice. arXiv:2301.01631.
[PA8FD08] Aldridge, Johnson, Scarlett. Group Testing: An Information Theory Perspective. arXiv:1902.06002.
[P06A777] Baldassini, Johnson, Aldridge. The Capacity of Adaptive Group Testing. arXiv:1301.7023.
[P9AD26E] Aldridge. The capacity of Bernoulli nonadaptive group testing. arXiv:1511.05201.
[PE5CF06] Cheraghchi. Noise-Resilient Group Testing: Limitations and Constructions. arXiv:0811.2609.
[P5E6A4C] Chan, Jaggi, Saligrama, Agrawal. Non-adaptive Group Testing: Explicit bounds and novel algorithms. arXiv:1202.0206.
[P08CCDD] Coja-Oghlan, Gebhard, Hahn-Klimroth, Loick. Optimal group testing. arXiv:1911.02287.
[P685226] Wu, Cheng, Yang. A 1.431-Competitive Algorithm for Combinatorial Group Testing. arXiv:2310.09320.
[PFB9004] Scheidweiler, Triesch. Upper and Lower Bounds for Competitive Group Testing. arXiv:2012.02630.
[P5B0ED3] Clarfeld, Eppstein. Group Testing for Efficiently Sampling Hypergraphs When Tests Have Variable Costs. arXiv:2010.09205.
[PCFF773] Knill. Lower bounds for identifying subset members with subset queries. arXiv:9411219.
[P320853] Shangguan, Ge. New bounds on the number of tests for disjunct matrices. arXiv:1501.03566.
[PD8BD7C] D'yachkov, Vorobyev, Polyanskii. Bounds on the rate of superimposed codes. arXiv:1401.0050.
[PEAF8CB] Jia, Navidi, Nagarajan. Optimal Decision Tree and Adaptive Submodular Ranking with Noisy Outcomes. arXiv:2312.15357.
[PC4925D] Saettler, Laber, Cicalese. Trading off Worst and Expected Cost in Decision Tree Problems and a Value Dependent Model. arXiv:1406.3655.
[P1A2D80] Bellala, Bhavnani, Scott. Query Learning with Exponential Query Costs. arXiv:1002.4019.
[PB950B4] Baer. On Conditional Branches in Optimal Decision Trees. arXiv:0611037.
[P95964D] Dereniowski, Kosowski, Uznański. Approximation Strategies for Generalized Binary Search in Weighted Trees. arXiv:1702.08207.
[P1FCE5B] Gunluk, Kalagnanam, Li, Menickelly, Wolfe. Optimal Generalized Decision Trees via Integer Programming. arXiv:1612.03225.
[PE18BCE] Li, Liang, Mussmann. A Tight Analysis of Greedy Yields Subexponential Time Approximation for Uniform Decision Tree. arXiv:1906.11385.
[P2914C1] van der Linden, de Weerdt, Demirović. Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming. arXiv:2305.19706.
[P4A3914] Turney. Cost-Sensitive Classification: Empirical Evaluation of a Hybrid Genetic Decision Tree Induction Algorithm. JAIR 2 (1995). arXiv:9503102.
[P6D10EE] Higger et al. Fast Switch Scanning Keyboards: Minimal Expected Query Decision Trees. arXiv:1606.02552.
[P396637] Nguyen. Optimal non-adaptive solutions for the counterfeit coin problem. arXiv:1502.04896.
[P34C7AC] Khovanova, Knop. Coins of Three Different Weights. arXiv:1409.0250.
[P3DC0F4] Khovanova, Knop, Polubasov. Chameleon Coins. arXiv:1512.07338.
[PB98308] Waghela. A unified picture of Balance puzzles and Group testing. arXiv:2108.02014.
[P215DAE] Mitzenmacher, Vassilvitskii. Algorithms with Predictions. arXiv:2006.09123.
[P23268F] Angelopoulos, Kamali, Zhang. Online Search With Best-Price and Query-Based Predictions. arXiv:2112.01592.
[PD4ACBA] Wang, Zhao, Chuah. Optimal Nested Test Plan for Combinatorial Quantitative Group Testing. arXiv:1407.2283.
