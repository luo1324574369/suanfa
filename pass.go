package main

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"
)

// 62. 不同路径
// https://leetcode.cn/problems/unique-paths/description/?envType=study-plan-v2&envId=dynamic-programming
func maxDiff(num int) int {
	maxNum, minNum := 0, math.MaxInt

	numStr := strconv.Itoa(num)
	var allMaxChange []int
	var allMinChange []int
	for i := 0; i < 9; i++ {
		n, _ := strconv.Atoi(strings.Replace(numStr, strconv.Itoa(i), "9", -1))
		allMaxChange = append(allMaxChange, n)
	}
	for i := 1; i < 10; i++ {
		if string(numStr[0]) == strconv.Itoa(i) {
			n, _ := strconv.Atoi(strings.Replace(numStr, strconv.Itoa(i), "1", -1))
			allMinChange = append(allMinChange, n)
			continue
		}
		n, _ := strconv.Atoi(strings.Replace(numStr, strconv.Itoa(i), "0", -1))
		allMinChange = append(allMinChange, n)
	}

	for i := 0; i < len(allMaxChange); i++ {
		if maxNum < allMaxChange[i] {
			maxNum = allMaxChange[i]
		}
	}
	for i := 0; i < len(allMinChange); i++ {
		if minNum > allMinChange[i] {
			minNum = allMinChange[i]
		}
	}
	return maxNum - minNum
}

// 3170. 删除星号以后字典序最小的字符串
// https://leetcode.cn/problems/lexicographically-minimum-string-after-removing-stars/description/?envType=daily-question&envId=2025-06-07
func clearStars(s string) string {
	slack := make([][]int, 26)
	for i := 0; i < 26; i++ {
		slack[i] = make([]int, 0)
	}

	bs := []byte(s)
	for i := 0; i < len(s); i++ {
		if s[i] != '*' {
			slack[s[i]-'a'] = append(slack[s[i]-'a'], i)
		} else {
			for j := 0; j < 26; j++ {
				if len(slack[j]) != 0 {
					last := slack[j][len(slack[j])-1]
					bs[last] = '*'
					slack[j] = slack[j][:len(slack[j])-1]
					break
				}
			}
		}
	}

	res := ""
	for i := 0; i < len(s); i++ {
		if bs[i] != '*' {
			res += string(bs[i])
		}
	}
	return res
}

// 2359. 找到离给定两个节点最近的节点
// https://leetcode.cn/problems/find-closest-node-to-given-two-nodes/description/
func closestMeetingNode(edges []int, node1 int, node2 int) int {
	lenEdges := len(edges)
	node1Distance := make([]int, lenEdges)
	node2Distance := make([]int, lenEdges)

	for i := 0; i < lenEdges; i++ {
		node1Distance[i] = -1
		node2Distance[i] = -1
	}

	countDistance(edges, node1, node1Distance)
	countDistance(edges, node2, node2Distance)

	minDistance := -1
	minNode := -1
	for i := 0; i < lenEdges; i++ {
		if node1Distance[i] != -1 && node2Distance[i] != -1 {
			currentMinDistance := max(node1Distance[i], node2Distance[i])
			if currentMinDistance < minDistance || minDistance == -1 {
				minDistance = currentMinDistance
				minNode = i
			} else if currentMinDistance == minDistance && i < minNode {
				minNode = i
			}
		}
	}

	return minNode
}

func countDistance(edges []int, startNode int, nodeDistance []int) {
	tempNode := startNode
	currentDistance := 0
	for nodeDistance[tempNode] == -1 {
		nodeDistance[tempNode] = currentDistance
		currentDistance++
		if edges[tempNode] == -1 {
			break
		}
		tempNode = edges[tempNode]
	}
}

// 2131. 连接两字母单词得到的最长回文串
// https://leetcode.cn/problems/longest-palindrome-by-concatenating-two-letter-words/description/?envType=daily-question&envId=2025-05-25
func longestPalindrome2(words []string) int {
	allWords := make(map[string]int)
	for i := 0; i < len(words); i++ {
		allWords[words[i]]++
	}

	res := 0
	mid := false
	for word, _ := range allWords {
		reverseWord := string(word[1]) + string(word[0])

		if reverseWord == word {
			if allWords[word]%2 == 1 {
				mid = true
			}
			res += (allWords[word] / 2) * 4
		} else {
			_, ok := allWords[reverseWord]
			if !ok {
				continue
			}
			res += min(allWords[word], allWords[reverseWord]) * 4
			delete(allWords, reverseWord)
		}
	}

	if mid {
		res += 2
	}

	return res
}

// 64. 最小路径和
// https://leetcode.cn/problems/minimum-path-sum/description/?envType=study-plan-v2&envId=dynamic-programming
func minPathSum(grid [][]int) int {
	m, n := len(grid), len(grid[0])

	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}

	dp[0][0] = grid[0][0]
	for i := 1; i < n; i++ {
		dp[0][i] = dp[0][i-1] + grid[0][i]
	}
	for i := 1; i < m; i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}

	for i := 1; i < m; i++ {
		for j := 1; j < n; j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}

	return dp[m-1][n-1]
}

// 198. 打家劫舍
// https://leetcode.cn/problems/house-robber/?envType=study-plan-v2&envId=dynamic-programming
func rob(nums []int) int {
	memo := make([]int, len(nums))
	for i := 0; i < len(nums); i++ {
		memo[i] = -1
	}
	return dpFunc(nums, memo, 0)
}

func dpFunc(nums []int, memo []int, index int) int {
	if index >= len(nums) {
		return 0
	}

	if memo[index] != -1 {
		return memo[index]
	}

	res := max(dpFunc(nums, memo, index+1), dpFunc(nums, memo, index+2)+nums[index])
	memo[index] = res
	return res
}

// 2900. 最长相邻不相等子序列 I
// https://leetcode.cn/problems/longest-unequal-adjacent-groups-subsequence-i/
func getLongestSubsequence(words []string, groups []int) []string {
	dp := make([]int, len(words))
	dpStr := make([][]string, len(words))
	for i := 0; i < len(words); i++ {
		dp[i] = 1
		dpStr[i] = []string{words[i]}
	}

	for i := 0; i < len(words); i++ {
		for j := 0; j < i; j++ {
			if groups[i] != groups[j] {
				if dp[j]+1 > dp[i] {
					dp[i] = dp[j] + 1
					// 创建一个新的切片并复制 dpStr[j] 的内容
					newSeq := make([]string, len(dpStr[j]))
					copy(newSeq, dpStr[j])
					// 追加当前元素
					dpStr[i] = append(newSeq, words[i])
				}
			}
		}
	}

	var res []string
	for i := 0; i < len(dpStr); i++ {
		if len(res) < len(dpStr[i]) {
			res = dpStr[i]
		}
	}
	return res
}

// 2962. 统计最大元素出现至少 K 次的子数组
// https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/?envType=daily-question&envId=2025-04-29
func countSubarrays(nums []int, k int) int64 {
	maxNum := 0
	for i := 0; i < len(nums); i++ {
		if nums[i] > maxNum {
			maxNum = nums[i]
		}
	}

	maxCount := 0
	left, right := 0, 0
	var res int64
	for right < len(nums) {
		if nums[right] == maxNum {
			maxCount++
		}
		right++

		for maxCount >= k {
			if nums[left] == maxNum {
				maxCount--
			}
			left++
		}
		res += int64(left)
	}
	return res
}

// 2799. 统计完全子数组的数目
// https://leetcode.cn/problems/count-complete-subarrays-in-an-array/submissions/625055083/?envType=daily-question&envId=2025-04-24
func countCompleteSubarrays(nums []int) int {
	diff := make(map[int]int)
	for _, v := range nums {
		diff[v]++
	}
	diffNum := len(diff)

	result := 0
	left, right := 0, 0
	windowDiff := make(map[int]int)
	for right < len(nums) {
		windowDiff[nums[right]]++
		right++

		for len(windowDiff) == diffNum {
			result += len(nums) - right + 1

			if windowDiff[nums[left]] == 1 {
				delete(windowDiff, nums[left])
			} else {
				windowDiff[nums[left]]--
			}
			left++
		}
	}
	return result
}

// 1399. 统计最大组的数目
// https://leetcode.cn/problems/count-largest-group/description/
func countLargestGroup(n int) int {
	maxNum := 0
	frenquence := make(map[int]int)

	for i := 1; i <= n; i++ {
		s := sumNum(i)
		frenquence[s]++
		if frenquence[s] > maxNum {
			maxNum = frenquence[s]
		}
	}

	result := 0
	for _, v := range frenquence {
		if v == maxNum {
			result++
		}
	}

	return result
}

func sumNum(num int) int {
	result := 0
	for num != 0 {
		result += num % 10
		num = num / 10
	}
	return result
}

// 2176. 统计数组中相等且可以被整除的数对
// https://leetcode.cn/problems/count-equal-and-divisible-pairs-in-an-array/description/?envType=daily-question&envId=2025-04-17
func countPairs(nums []int, k int) int {
	res := 0
	for i := 0; i < len(nums); i++ {
		backtrap(nums, i, []int{}, k, &res)
	}
	return res
}

func backtrap(nums []int, start int, path []int, k int, res *int) {
	if len(path) == 1 {
		*res++
		return
	}

	for i := start + 1; i < len(nums); i++ {
		if (start*i)%k != 0 {
			continue
		}
		if nums[start] != nums[i] {
			continue
		}

		path = append(path, i)
		backtrap(nums, start, path, k, res)
		path = path[:len(path)-1]
	}
}

// 2537. 统计好子数组的数目
// https://leetcode.cn/problems/count-the-number-of-good-subarrays/description/
func countGood(nums []int, k int) int64 {
	n := len(nums)
	same := 0 // 存储符合 满足 i < j 且 arr[i] == arr[j] 的数量
	var ans int64 = 0
	left, right := 0, 0
	cnt := make(map[int]int) // 存储每个数字的出现次数

	for right < n {
		same += cnt[nums[right]]
		cnt[nums[right]]++

		for same >= k {
			ans += int64(n - right)

			cnt[nums[left]]--
			same -= cnt[nums[left]]
			left++
		}
		right++
	}

	// same, right := 0, -1
	// cnt := make(map[int]int)
	// var ans int64 = 0
	// for left := 0; left < n; left++ {
	//     for same < k && right + 1 < n {
	//         right++
	//         same += cnt[nums[right]]
	//         cnt[nums[right]]++
	//     }
	//     if same >= k {
	//         ans += int64(n - right)
	//     }
	//     cnt[nums[left]]--
	//     same -= cnt[nums[left]]
	// }
	return ans
}

// 11. 盛最多水的容器
// https://leetcode.cn/problems/container-with-most-water/description/
func maxArea(height []int) int {
	var result int
	left := 0
	right := len(height) - 1

	for left < right {
		temp := min(height[left], height[right]) * (right - left)
		if temp > result {
			result = temp
		}

		if height[left] > height[right] {
			right--
		} else {
			left++
		}
	}

	return result
}

// 528. 按权重随机选择
// https://leetcode.cn/problems/random-pick-with-weight/submissions/619349703/
type Solution5 struct {
	preSum []int
}

func Constructor5(w []int) Solution5 {
	preSum := make([]int, len(w)+1)

	for i := 0; i < len(w); i++ {
		preSum[i+1] = preSum[i] + w[i]
	}

	return Solution5{
		preSum: preSum,
	}
}

func (this *Solution5) PickIndex() int {
	max := this.preSum[len(this.preSum)-1]
	target := rand.Intn(max) + 1
	return leftBound(this.preSum, target) - 1
}

func leftBound5(nums []int, target int) int {
	left, right := 0, len(nums)-1

	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		}
	}
	return left
}

// 310. 最小高度树
// https://leetcode.cn/problems/minimum-height-trees/
func findMinHeightTrees(n int, edges [][]int) []int {
	if n == 1 {
		// base case，只有一个节点 0 的话，无法形成边，所以直接返回节点 0
		return []int{0}
	}

	gragh := make([][]int, n)

	for _, v := range edges {
		gragh[v[0]] = append(gragh[v[0]], v[1])
		gragh[v[1]] = append(gragh[v[1]], v[0])
	}

	var q []int
	for k, v := range gragh {
		if len(v) == 1 {
			q = append(q, k)
		}
	}

	curCount := n
	for curCount > 2 {
		qSize := len(q)
		for i := 0; i < qSize; i++ {
			cur := q[0]
			q = q[1:]
			curCount--

			for _, neighbor := range gragh[cur] {
				for kk, vv := range gragh[neighbor] {
					if vv == cur {
						gragh[neighbor] = append(gragh[neighbor][:kk], gragh[neighbor][kk+1:]...)
					}
				}

				if len(gragh[neighbor]) == 1 {
					q = append(q, neighbor)
				}
			}

			gragh[cur] = []int{}
		}
	}

	return q
}

// 1593. 拆分字符串使唯一子字符串的数目最大
// https://leetcode.cn/problems/split-a-string-into-the-max-number-of-unique-substrings/submissions/614364689/
func maxUniqueSplit(s string) int {
	set := make(map[string]bool)
	var backtrack func(string, int)
	res := 0

	backtrack = func(s string, index int) {
		if index == len(s) {
			// Calculate the maximum number of unique substrings
			if len(set) > res {
				res = len(set)
			}
			return
		}

		for i := index; i < len(s); i++ {
			sub := s[index : i+1]
			if set[sub] {
				continue
			}

			set[sub] = true
			backtrack(s, i+1)
			delete(set, sub)
		}

		// // 选择不切，什么都不做，直接走到下一个索引空隙
		// backtrack(s, index+1)

		// // 选择切，把 s[0..index] 切分出来作为一个子串
		// sub := s[:index+1]
		// // 按照题目要求，不能有重复的子串
		// if !set[sub] {
		//     // 做选择
		//     set[sub] = true

		//     // 剩下的字符继续穷举
		//     backtrack(s[index+1:], 0)

		//     // 撤销选择
		//     delete(set, sub)
		// }
	}

	backtrack(s, 0)
	return res
}

// 93. 复原 IP 地址
// https://leetcode.cn/problems/restore-ip-addresses/submissions/614270541/
func restoreIpAddresses(s string) []string {
	var res []string
	backtrack(s, 0, []string{}, &res)
	return res
}

func backtrack(s string, start int, path []string, res *[]string) {
	if start >= len(s) && len(path) == 4 {
		temp := strings.Join(path, ".")
		*res = append(*res, temp)
		return
	}

	for i := start; i < len(s); i++ {
		if !isValid3(s[start : i+1]) {
			continue
		}

		path = append(path, s[start:i+1])
		backtrack(s, i+1, path, res)
		path = path[:len(path)-1]
	}
}

func isValid3(s string) bool {
	if len(s) > 1 && s[0] == '0' {
		return false
	}
	i, _ := strconv.Atoi(s)
	if i < 0 || i > 255 {
		return false
	}
	return true
}

// 37. 解数独
// https://leetcode.cn/problems/sudoku-solver/submissions/611631707/
func solveSudoku(board [][]byte) {
	isFound := false
	var backtrack func(board [][]byte, index int)
	backtrack = func(board [][]byte, index int) {
		m, n := 9, 9
		if index == m*n {
			isFound = true
			return
		}

		if isFound {
			return
		}

		r, c := index/9, index%9

		if board[r][c] != '.' {
			backtrack(board, index+1)
			return
		}

		for i := 1; i <= 9; i++ {
			if !isValid2(board, r, c, i) {
				continue
			}

			board[r][c] = byte('0' + i)
			backtrack(board, index+1)
			if isFound {
				return
			}
			board[r][c] = '.'
		}
	}

	backtrack(board, 0)
}

func isValid2(board [][]byte, r, c, num int) bool {
	for i := 0; i < 9; i++ {
		if board[r][i] == byte('0'+num) {
			return false
		}

		if board[i][c] == byte('0'+num) {
			return false
		}

		if board[r/3*3+i/3][c/3*3+i%3] == byte('0'+num) {
			return false
		}
	}
	return true
}

// 743. 网络延迟时间
// https://leetcode.cn/problems/network-delay-time/
func networkDelayTime(times [][]int, n int, k int) int {
	graph := make([]map[int]int, n+1)
	for i := 0; i < n+1; i++ {
		graph[i] = make(map[int]int)
	}
	for _, edge := range times {
		graph[edge[0]][edge[1]] = edge[2]
	}

	distTo := dijkstra(k, graph)
	res := 0
	for i := 1; i < len(distTo); i++ {
		if distTo[i] == math.MaxInt64 {
			return -1
		}
		res = max(res, distTo[i])
	}
	return res
}

// 1584. 连接所有点的最小费用
// https://leetcode.cn/problems/min-cost-to-connect-all-points/description/
func minCostConnectPoints(points [][]int) int {
	var edges [][]int

	l := len(points)
	for i := 0; i < l; i++ {
		for j := i + 1; j < l; j++ {
			xi, yi := points[i][0], points[i][1]
			xj, yj := points[j][0], points[j][1]

			edges = append(edges, []int{i, j, int(math.Abs(float64(xi)-float64(xj))) + int(math.Abs(float64(yi)-float64(yj)))})
		}
	}

	sort.Slice(edges, func(a, b int) bool {
		return edges[a][2] < edges[b][2]
	})

	parent := make([]int, l)
	for i := 0; i < l; i++ {
		parent[i] = i
	}

	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}

	union := func(x, y int) {
		parentX := find(x)
		parentY := find(y)

		if parentX == parentY {
			return
		}

		parent[parentX] = parentY
	}

	isConnect := func(x, y int) bool {
		return find(x) == find(y)
	}

	mst := 0
	for _, edge := range edges {
		if isConnect(edge[0], edge[1]) {
			continue
		}
		mst += edge[2]
		union(edge[0], edge[1])
	}
	return mst
}

// 990. 等式方程的可满足性
// https://leetcode.cn/problems/satisfiability-of-equality-equations/submissions/609863133/
func equationsPossible(equations []string) bool {
	parent := make(map[int]int, 26)
	for i := 0; i < 3; i++ {
		parent[i] = i
	}

	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}

	union := func(x, y int) {
		parentX := find(x)
		parentY := find(y)

		if parentX == parentY {
			return
		}
		parent[parentX] = parentY
	}

	for _, eq := range equations {
		if eq[1] == '=' {
			union(int(eq[0]-'a'), int(eq[3]-'a'))
		}
	}

	for _, eq := range equations {
		if eq[1] == '!' {
			parentA := find(int(eq[0] - 'a'))
			parentB := find(int(eq[3] - 'a'))
			if parentA == parentB {
				return false
			}
		}
	}
	return true
}

// 684. 冗余连接
// https://leetcode.cn/problems/redundant-connection/description/
func findRedundantConnection(edges [][]int) []int {
	parent := make([]int, len(edges)+1)
	for i := range edges {
		parent[i] = i
	}

	var find func(x int) int
	find = func(x int) int {
		if parent[x] != x {
			parent[x] = find(parent[x])
		}
		return parent[x]
	}
	union := func(x, y int) bool {
		parentX := find(x)
		parentY := find(y)

		if parentX == parentY {
			return false
		}
		parent[parentX] = parentY
		return true
	}
	for _, edge := range edges {
		if !union(edge[0], edge[1]) {
			return edge
		}
	}
	return nil
}

// 785. 判断二分图
// https://leetcode.cn/problems/is-graph-bipartite/submissions/606683511/
func isBipartite(graph [][]int) bool {
	res := true
	viststed := make(map[int]int)

	for i := 0; i < len(graph); i++ {
		if !res {
			break
		}
		traverse2(graph, i, 1, &viststed, &res)
	}

	return res
}

func traverse2(graph [][]int, v int, vFlag int, viststed *map[int]int, res *bool) {
	if (*viststed)[v] != 0 {
		return
	}

	(*viststed)[v] = vFlag
	for _, vv := range graph[v] {
		if (*viststed)[vv] == 0 {
			traverse2(graph, vv, -vFlag, viststed, res)
		} else {
			if (*viststed)[v]+(*viststed)[vv] != 0 {
				*res = false
				return
			}
		}
	}
}

// 950. 按递增顺序显示卡牌
// https://leetcode.cn/problems/reveal-cards-in-increasing-order/description/
func deckRevealedIncreasing(deck []int) []int {
	sort.Ints(deck)

	var res []int
	for i := len(deck) - 1; i >= 0; i-- {
		if len(res) > 0 {
			last := res[len(res)-1]
			res = res[:len(res)-1]
			res = append([]int{last}, res...)
		}
		res = append([]int{deck[i]}, res...)
	}

	return res
}

// 450. 删除二叉搜索树中的节点
// https://leetcode.cn/problems/delete-node-in-a-bst/submissions/604967098/
func deleteNode(root *TreeNode, key int) *TreeNode {
	return deleteN(root, key)
}

func deleteN(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return root
	}

	root.Left = deleteN(root.Left, key)
	root.Right = deleteN(root.Right, key)

	if key != root.Val {
		return root
	}

	if root.Left == nil && root.Right == nil {
		return nil
	}

	if root.Left == nil {
		return root.Right
	}

	if root.Right == nil {
		return root.Left
	}

	min := getMin(root.Right)
	root.Val = min.Val
	root.Right = deleteN(root.Right, min.Val)
	return root
}

func getMin(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	if root.Left == nil {
		return root
	}
	return getMin(root.Left)
}

// 98. 验证二叉搜索树
// https://leetcode.cn/problems/validate-binary-search-tree/description/
func isValidBST(root *TreeNode) bool {
	maxVal := math.MinInt
	return traverse(root, &maxVal)
}

func traverse(root *TreeNode, maxVal *int) bool {
	if root == nil {
		return true
	}

	if traverse(root.Left, maxVal) == false {
		return false
	}

	if root.Val <= *maxVal {
		return false
	}
	*maxVal = root.Val

	if traverse(root.Right, maxVal) == false {
		return false
	}

	return true
}

// 654. 最大二叉树
// https://leetcode.cn/problems/maximum-binary-tree/description/
func constructMaximumBinaryTree2(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	return build(nums, 0, len(nums)-1)
}

func build(nums []int, left, right int) *TreeNode {
	if left > right {
		return nil
	}

	maxIndex := findMax(nums, left, right)
	root := &TreeNode{Val: nums[maxIndex]}

	if maxIndex-left > 0 {
		left := build(nums, left, maxIndex-1)
		root.Left = left
	}
	if right-maxIndex > 0 {
		right := build(nums, maxIndex+1, right)
		root.Right = right
	}

	return root
}

func findMax(nums []int, left, right int) int {
	max := nums[left]
	maxIndex := left
	for i := left + 1; i <= right; i++ {
		if nums[i] > max {
			max = nums[i]
			maxIndex = i
		}
	}
	return maxIndex
}

// 894. 所有可能的真二叉树
// https://leetcode.cn/problems/all-possible-full-binary-trees/
func allPossibleFBT(n int) []*TreeNode {
	s := &Solution2{
		memo: make(map[int][]*TreeNode),
	}
	return s.allPossibleFBT(n)
}

type Solution2 struct {
	memo map[int][]*TreeNode
}

func (s *Solution2) allPossibleFBT(n int) []*TreeNode {
	if n%2 == 0 {
		return []*TreeNode{}
	}
	return s.build(n)
}

func (s *Solution2) build(n int) []*TreeNode {
	if n == 1 {
		return []*TreeNode{
			&TreeNode{},
		}
	}

	if v, ok := s.memo[n]; ok {
		return v
	}

	var res []*TreeNode
	for i := 1; i < n; i += 2 {
		j := n - i - 1

		left := s.build(i)
		right := s.build(j)

		for _, l := range left {
			for _, r := range right {
				root := &TreeNode{}
				root.Left = l
				root.Right = r
				res = append(res, root)
			}
		}
	}

	s.memo[n] = res
	return res
}

// 129. 求根节点到叶节点数字之和
// https://leetcode.cn/problems/sum-root-to-leaf-numbers/description/
func binaryTreePaths(root *TreeNode) []string {
	var res []string
	if root == nil {
		return res
	}
	addPathToleaf(root, []string{}, &res)
	return res
}

func addPathToleaf(root *TreeNode, path []string, res *[]string) {
	if root == nil {
		return
	}
	if root.Left == nil && root.Right == nil {
		path = append(path, strconv.Itoa(root.Val))
		*res = append(*res, strings.Join(path, "->"))
		return
	}
	path = append(path, strconv.Itoa(root.Val))
	if root.Left != nil {
		addPathToleaf(root.Left, path, res)
	}
	if root.Right != nil {
		addPathToleaf(root.Right, path, res)
	}
}

// 1670. 设计前中后队列
// https://leetcode.cn/problems/design-front-middle-back-queue/
type FrontMiddleBackQueue struct {
	elements []int
}

func Constructor4() FrontMiddleBackQueue {
	return FrontMiddleBackQueue{}
}

func (this *FrontMiddleBackQueue) PushFront(val int) {
	if len(this.elements) == 0 {
		this.PushBack(val)
		return
	}
	this.elements = append([]int{val}, this.elements...)
}

func (this *FrontMiddleBackQueue) PushMiddle(val int) {
	if len(this.elements) == 0 {
		this.PushBack(val)
		return
	}
	mid := len(this.elements) / 2
	tail := make([]int, len(this.elements[mid:]))
	copy(tail, this.elements[mid:])
	this.elements = append(this.elements[:mid], val)
	this.elements = append(this.elements[:mid+1], tail...)
}

func (this *FrontMiddleBackQueue) PushBack(val int) {
	this.elements = append(this.elements, val)
}

func (this *FrontMiddleBackQueue) PopFront() int {
	if len(this.elements) == 0 {
		return -1
	}
	front := this.elements[0]
	this.elements = this.elements[1:]
	return front
}

func (this *FrontMiddleBackQueue) PopMiddle() int {
	if len(this.elements) <= 1 {
		return this.PopFront()
	}
	midIndex := (len(this.elements) / 2)
	if len(this.elements)%2 == 0 {
		midIndex--
	}
	mid := this.elements[midIndex]
	this.elements = append(this.elements[:midIndex], this.elements[midIndex+1:]...)
	return mid
}

func (this *FrontMiddleBackQueue) PopBack() int {
	if len(this.elements) <= 1 {
		return this.PopFront()
	}
	tail := this.elements[len(this.elements)-1]
	this.elements = this.elements[:len(this.elements)-1]
	return tail
}

/**
 * Your FrontMiddleBackQueue object will be instantiated and called as such:
 * obj := Constructor();
 * obj.PushFront(val);
 * obj.PushMiddle(val);
 * obj.PushBack(val);
 * param_4 := obj.PopFront();
 * param_5 := obj.PopMiddle();
 * param_6 := obj.PopBack();
 */

// 641. 设计循环双端队列
// https://leetcode.cn/problems/design-circular-deque/submissions/602125330/
type MyCircularDeque struct {
	head  int
	tail  int
	queue []int
}

func Constructor2(k int) MyCircularDeque {
	return MyCircularDeque{
		queue: make([]int, k+1),
	}
}

func (this *MyCircularDeque) InsertFront(value int) bool {
	if this.IsFull() {
		return false
	}
	this.head = (this.head - 1 + len(this.queue)) % len(this.queue)
	this.queue[this.head] = value
	return true
}

func (this *MyCircularDeque) InsertLast(value int) bool {
	if this.IsFull() {
		return false
	}
	this.queue[this.tail] = value
	this.tail = (this.tail + 1) % len(this.queue)
	return true
}

func (this *MyCircularDeque) DeleteFront() bool {
	if this.IsEmpty() {
		return false
	}
	this.head = (this.head + 1) % len(this.queue)
	this.queue[this.head] = 0
	return true
}

func (this *MyCircularDeque) DeleteLast() bool {
	if this.IsEmpty() {
		return false
	}
	this.tail = (this.tail - 1 + len(this.queue)) % len(this.queue)
	this.queue[this.tail] = 0
	return true
}

func (this *MyCircularDeque) GetFront() int {
	return this.queue[this.head]
}

func (this *MyCircularDeque) GetRear() int {
	return this.queue[(this.tail-1+len(this.queue))%len(this.queue)]
}

func (this *MyCircularDeque) IsEmpty() bool {
	return this.tail == this.head
}

func (this *MyCircularDeque) IsFull() bool {
	return (this.tail+1)%len(this.queue) == this.head
}

/**
 * Your MyCircularDeque object will be instantiated and called as such:
 * obj := Constructor(k);
 * param_1 := obj.InsertFront(value);
 * param_2 := obj.InsertLast(value);
 * param_3 := obj.DeleteFront();
 * param_4 := obj.DeleteLast();
 * param_5 := obj.GetFront();
 * param_6 := obj.GetRear();
 * param_7 := obj.IsEmpty();
 * param_8 := obj.IsFull();
 */

// 388. 文件的最长绝对路径
// https://leetcode.cn/problems/longest-absolute-file-path/submissions/601484357/
func lengthLongestPath(input string) int {
	var stack []string

	parts := strings.Split(input, "\n")
	maxLength := 0

	for i := 0; i < len(parts); i++ {
		part := parts[i]
		level := strings.LastIndex(part, "\t") + 1

		for len(stack) > level {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, part[level:])

		if strings.Contains(part, ".") {
			currentLength := 0
			for j := 0; j < len(stack); j++ {
				currentLength += len(stack[j])
				currentLength += 1
			}
			currentLength -= 1
			if currentLength > maxLength {
				maxLength = currentLength
			}
		}

	}
	return maxLength
}

// 143. 重排链表
// https://leetcode.cn/problems/reorder-list/description/
func reorderList(head *ListNode) {
	var stack []*ListNode
	p := head
	for p != nil {
		stack = append(stack, p)
		p = p.Next
	}

	p = head
	for p != nil {
		lastNode := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		next := p.Next
		if lastNode.Next == next || lastNode == next {
			lastNode.Next = nil
			break
		}
		p.Next = lastNode
		lastNode.Next = next
		p = next
	}
}

// 71. 简化路径
// https://leetcode.cn/problems/simplify-path/description/
func simplifyPath(path string) string {
	var stack []string
	var cache string

	for i := 0; i < len(path); i++ {
		c := path[i]
		if c != '/' {
			cache += string(c)
			continue
		}
		if cache == "." {
			cache = ""
		}
		if cache == ".." {
			if len(stack) > 2 {
				stack = stack[:len(stack)-2]
			} else {
				stack = []string{"/"}
			}
			cache = ""
		}
		if cache != "" {
			stack = append(stack, string(cache))
			cache = ""
		}
		if len(stack) == 0 || stack[len(stack)-1] != "/" {
			stack = append(stack, string(c))
		}
	}
	if len(cache) != 0 {
		if cache == "." {
			cache = ""
		}
		if cache == ".." {
			if len(stack) > 2 {
				stack = stack[:len(stack)-2]
			} else {
				stack = []string{"/"}
			}
			cache = ""
		}
		if cache != "" {
			stack = append(stack, string(cache))
			cache = ""
		}
	}
	res := ""
	if stack[0] != "/" {
		res += "/"
	}
	if len(stack) > 1 && stack[len(stack)-1] == "/" {
		stack = stack[:len(stack)-1]
	}
	for i := 0; i < len(stack); i++ {
		res += stack[i]
	}
	return res
}

// 373. 查找和最小的 K 对数字
// https://leetcode.cn/problems/find-k-pairs-with-smallest-sums/description/
func kSmallestPairs(nums1 []int, nums2 []int, k int) [][]int {
	p := &PriorityQueue{}
	heap.Init(p)

	for i := 0; i < len(nums1); i++ {
		heap.Push(p, []int{nums1[i], nums2[0], 0})
	}

	ik := 0
	var res [][]int
	for p.Len() > 0 && ik < k {
		current := heap.Pop(p).([]int)
		ik++
		res = append(res, []int{current[0], current[1]})

		nextIndex := current[2] + 1
		if nextIndex < len(nums2) {
			heap.Push(p, []int{current[0], nums2[nextIndex], nextIndex})
		}
	}
	return res
}

// 707. 设计链表
// https://leetcode.cn/problems/design-linked-list/
// type MyLinkedList struct {
// 	preHead *Node // next指向head节点
// 	tail    *Node // 实际的tail节点
// 	size    int
// }

// type Node struct {
// 	Val  int
// 	Next *Node
// }

// func Constructor() MyLinkedList {
// 	head := &Node{}
// 	return MyLinkedList{preHead: head, tail: head}
// }

// func (this *MyLinkedList) Get(index int) int {
// 	if index < 0 || index >= this.size {
// 		return -1
// 	}
// 	pre := this.preHead
// 	for i := 0; i < index; i++ {
// 		pre = pre.Next
// 	}
// 	current := pre.Next
// 	return current.Val
// }

// func (this *MyLinkedList) AddAtHead(val int) {
// 	valNode := &Node{Val: val}
// 	valNode.Next = this.preHead.Next
// 	this.preHead.Next = valNode
// 	if this.size == 0 {
// 		this.tail = valNode
// 	}
// 	this.size++
// }

// func (this *MyLinkedList) AddAtTail(val int) {
// 	valNode := &Node{Val: val}
// 	this.tail.Next = valNode
// 	this.tail = valNode
// 	this.size++
// }

// func (this *MyLinkedList) AddAtIndex(index int, val int) {
// 	if index < 0 || index > this.size {
// 		return
// 	}
// 	if index == this.size {
// 		this.AddAtTail(val)
// 		return
// 	}
// 	pre := this.preHead
// 	for i := 0; i < index; i++ {
// 		pre = pre.Next
// 	}
// 	valNode := &Node{Val: val}
// 	valNode.Next = pre.Next
// 	pre.Next = valNode
// 	this.size++
// }

// func (this *MyLinkedList) DeleteAtIndex(index int) {
// 	if index < 0 || index >= this.size {
// 		return
// 	}
// 	pre := this.preHead
// 	for i := 0; i < index; i++ {
// 		pre = pre.Next
// 	}
// 	current := pre.Next
// 	pre.Next = current.Next
// 	if index == this.size-1 {
// 		this.tail = pre
// 	}
// 	this.size--
// }

// 518. 零钱兑换 II
// https://leetcode.cn/problems/coin-change-ii/submissions/
func change(amount int, coins []int) int {
	coinsLen := len(coins)
	dp := make(map[int][]int, coinsLen)
	for i := 0; i <= coinsLen; i++ {
		dp[i] = make([]int, amount+1)
		dp[i][0] = 1
	}

	for i := 1; i <= coinsLen; i++ {
		for j := 1; j <= amount; j++ {
			if j-coins[i-1] < 0 {
				dp[i][j] = dp[i-1][j]
			} else {
				dp[i][j] = dp[i-1][j] + dp[i][j-coins[i-1]]
			}
		}
	}
	return dp[coinsLen][amount]
}

// 1372. 二叉树中的最长交错路径
// https://leetcode.cn/problems/longest-zigzag-path-in-a-binary-tree/
func longestZigZag(root *TreeNode) int {
	result := 0
	deepPathLongestZigZag(root, 0, 0, &result)
	return result
}

func deepPathLongestZigZag(root *TreeNode, rightValue, leftValue int, result *int) {
	if root == nil {
		return
	}

	if rightValue > *result {
		*result = rightValue
	}

	if leftValue > *result {
		*result = leftValue
	}

	if root.Left != nil {
		deepPathLongestZigZag(root.Left, leftValue+1, 0, result)
	}

	if root.Right != nil {
		deepPathLongestZigZag(root.Right, 0, rightValue+1, result)
	}
}

// 437. 路径总和 III
// https://leetcode.cn/problems/path-sum-iii/
func pathSum(root *TreeNode, targetSum int) int {
	if root == nil {
		return 0
	}
	var result int
	deepPath(root, targetSum, &result)
	return result
}

func deepPath(root *TreeNode, targetSum int, result *int) {
	if root == nil {
		return
	}
	deepSum(root, targetSum, result)
	if root.Left != nil {
		deepPath(root.Left, targetSum, result)
	}
	if root.Right != nil {
		deepPath(root.Right, targetSum, result)
	}
}

func deepSum(root *TreeNode, target int, result *int) {
	if root == nil {
		return
	}
	if target-root.Val == 0 {
		*result++
	}
	if root.Left != nil {
		deepSum(root.Left, target-root.Val, result)
	}
	if root.Right != nil {
		deepSum(root.Right, target-root.Val, result)
	}
	return
}

// 394. 字符串解码
// https://leetcode.cn/problems/decode-string/
func decodeString(s string) string {
	return decode(s, 1)
}

func decode(s string, times int) string {
	currentTimes := 0
	subIndex := -1
	index := 0

	for index < len(s) {
		for '0' <= s[index] && s[index] <= '9' {
			if subIndex == -1 {
				subIndex = index
			}
			currentTimes = currentTimes*10 + int(s[index]-'0')
			index++
		}
		if s[index] == '[' {
			s = s[0:subIndex] + decode(s[index+1:], currentTimes)
			index = subIndex
			subIndex = -1
			currentTimes = 0
		}
		if s[index] == ']' {
			break
		}
		index++
	}

	if subIndex == -1 {
		subIndex = 0
	}
	if index == len(s) {
		return strings.Repeat(s[:index], times)
	}
	return strings.Repeat(s[subIndex:index], times) + s[index+1:]
}

// 124. 二叉树中的最大路径和
// https://leetcode.cn/problems/binary-tree-maximum-path-sum/submissions/
func maxPathSum(root *TreeNode) int {
	res := math.MinInt
	var deepMaxPathSum func(root *TreeNode) int
	deepMaxPathSum = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		v := root.Val
		leftValue, rightValue := 0, 0
		if root.Left != nil {
			leftValue = max(deepMaxPathSum(root.Left), 0)
		}
		if root.Right != nil {
			rightValue = max(deepMaxPathSum(root.Right), 0)
		}

		res = max(v+leftValue+rightValue, res)
		return max(v+rightValue, v+leftValue)
	}
	deepMaxPathSum(root)
	return res
}

// 496. 下一个更大元素 I
// https://leetcode.cn/problems/next-greater-element-i/submissions/
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	greater := nextGreaterElements(nums2)
	greaterMap := make(map[int]int)
	for i := 0; i < len(nums2); i++ {
		greaterMap[nums2[i]] = greater[i]
	}

	var result []int
	for i := 0; i < len(nums1); i++ {
		result = append(result, greaterMap[nums1[i]])
	}
	return result
}

func nextGreaterElements(nums []int) []int {
	var stack []int
	result := make([]int, len(nums))

	for i := len(nums) - 1; i >= 0; i-- {
		for len(stack) > 0 && nums[i] >= stack[len(stack)-1] {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 {
			result[i] = -1
		} else {
			result[i] = stack[len(stack)-1]
		}
		stack = append(stack, nums[i])
	}
	return result
}

// 494. 目标和
// https://leetcode.cn/problems/target-sum/submissions/
func findTargetSumWays(nums []int, target int) int {
	memo := make(map[string]int)
	return dpFindTargetSumWays(nums, target, 0, 0, memo)
}

func dpFindTargetSumWays(nums []int, target int, index int, sum int, memo map[string]int) int {
	if index == len(nums) {
		if sum == target {
			return 1
		} else {
			return 0
		}
	}
	key := strconv.Itoa(index) + "," + strconv.Itoa(target-sum)
	if _, ok := memo[key]; ok {
		return memo[key]
	}
	dpResult := dpFindTargetSumWays(nums, target, index+1, sum+nums[index], memo) + dpFindTargetSumWays(nums, target, index+1, sum-nums[index], memo)
	memo[key] = dpResult
	return dpResult
}

// 72. 编辑距离
// https://leetcode.cn/problems/edit-distance/description/
func minDistance(word1 string, word2 string) int {
	word1Len := len(word1)
	word2Len := len(word2)

	dp := make(map[int][]int, word1Len+1)
	for i := 0; i <= word1Len; i++ {
		dp[i] = make([]int, word2Len+1)
		dp[i][0] = i
	}

	for i := 0; i <= word2Len; i++ {
		dp[0][i] = i
	}

	for i := 1; i <= word1Len; i++ {
		for j := 1; j <= word2Len; j++ {
			if word1[i-1] == word2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min3(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)
			}
		}
	}

	return dp[word1Len][word2Len]
}

func min3(a, b, c int) int {
	return min2(a, min2(b, c))
}

func min2(a, b int) int {
	if a > b {
		return b
	}
	return a
}

// 139. 单词拆分
// https://leetcode.cn/problems/word-break/submissions/
func wordBreak(s string, wordDict []string) bool {
	memo := make(map[int]bool)
	return dp(s, wordDict, 0, memo)
}

func dp(target string, wordDict []string, index int, memo map[int]bool) bool {
	targetLen := len(target)
	if index == targetLen {
		return true
	}
	if index >= targetLen {
		return false
	}
	for _, w := range wordDict {
		if index+len(w) > targetLen {
			continue
		}
		if target[index:index+len(w)] != w {
			continue
		}
		if _, ok := memo[index]; ok {
			return memo[index]
		}
		subDp := dp(target, wordDict, index+len(w), memo)
		if subDp {
			memo[index] = true
			return true
		}
	}
	memo[index] = false
	return false
}

// 160. 相交链表
// https://leetcode.cn/problems/intersection-of-two-linked-lists/
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	p1 := headA
	p2 := headB
	for p1 != p2 {
		if p1 != nil {
			p1 = p1.Next
		} else {
			p1 = headB
		}
		if p2 != nil {
			p2 = p2.Next
		} else {
			p2 = headA
		}
	}
	return p1
}

// 1071. 字符串的最大公因子
// https://leetcode.cn/problems/greatest-common-divisor-of-strings/
func gcdOfStrings(str1 string, str2 string) string {
	shortStr := str1
	if len(str2) > len(str1) {
		shortStr = str1
	} else {
		shortStr = str2
	}

	for i := len(shortStr); i >= 1; i-- {
		tempStr := str1[0:i]
		isSubStr1 := isSubStr(tempStr, str1)
		isSubStr2 := isSubStr(tempStr, str2)
		if isSubStr1 && isSubStr2 {
			return tempStr
		}
	}
	return ""
}

func isSubStr(str1 string, str2 string) bool {
	str1Len := len(str1)
	str2Len := len(str2)
	if str2Len%str1Len != 0 || str2Len < str1Len {
		return false
	}
	i := 0
	for {
		index := str1Len * i
		if index+str1Len > str2Len {
			break
		}
		if str1 != str2[index:index+str1Len] {
			return false
		}
		i++
	}
	return true
}

// 34. 在排序数组中查找元素的第一个和最后一个位置
// https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/submissions/
func searchRange(nums []int, target int) []int {
	nl := len(nums)
	if nl == 0 {
		return []int{-1, -1}
	}

	leftB, rightB := -1, -1
	left, right := 0, nl-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			right = mid - 1
		} else if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		}
	}

	if left >= nl {
		return []int{-1, -1}
	}

	if nums[left] != target {
		return []int{-1, -1}
	}

	leftB = left

	left, right = 0, nl-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			left = mid + 1
		} else if nums[mid] > target {
			right = mid - 1
		} else if nums[mid] < target {
			left = mid + 1
		}
	}

	if left == 0 {
		return []int{-1, -1}
	}

	if nums[left-1] != target {
		return []int{-1, -1}
	}

	rightB = left - 1

	return []int{leftB, rightB}
}

// 105. 从前序与中序遍历序列构造二叉树
// https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
func buildTree(preorder []int, inorder []int) *TreeNode {

	preL := len(preorder)
	inL := len(inorder)

	if preL == 0 || inL == 0 {
		return nil
	}

	root := &TreeNode{Val: preorder[0]}
	i := 0
	for ; i < preL; i++ {
		if preorder[0] == inorder[i] {
			break
		}
	}
	leftSize := i

	root.Left = buildTree(preorder[1:1+leftSize], inorder[:i])
	root.Right = buildTree(preorder[1+leftSize:], inorder[i+1:])

	return root
}

// 698. 划分为k个相等的子集
// https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/
func canPartitionKSubsets(nums []int, k int) bool {
	used := 0
	hashMap := make(map[int]bool)
	lNums := len(nums)
	var sum int
	for _, num := range nums {
		sum += num
	}
	target := float64(sum) / float64(k)

	var backtrack func(start int, buketCurrentValue int, kNum int) bool
	backtrack = func(start int, buketCurrentValue int, kNum int) bool {
		if kNum == 0 {
			// 箱子为空,证明刚刚好
			return true
		}
		if float64(buketCurrentValue) == target {
			// 满足箱子,走下一个循环
			kNum--
			res := backtrack(0, 0, kNum)
			hashMap[used] = res
			return res
		}

		if r, ok := hashMap[used]; ok {
			return r
		}

		for i := start; i < lNums; i++ {
			if float64(buketCurrentValue+nums[i]) > target {
				continue
			}
			// 已经被使用过
			if (used >> i & 1) == 1 {
				continue
			}
			used |= 1 << i
			if backtrack(i+1, buketCurrentValue+nums[i], kNum) {
				return true
			}
			used ^= 1 << i
		}
		return false
	}

	return backtrack(0, 0, k)
}

// 567. 字符串的排列
// https://leetcode.cn/problems/permutation-in-string/
func checkInclusion(s1 string, s2 string) bool {
	valid := 0
	left := 0
	right := 0
	s1l := len(s1)
	need := make(map[uint8]int)
	window := make(map[uint8]int)
	for i := 0; i < s1l; i++ {
		need[s1[i]]++
	}

	for right < len(s2) {
		c := s2[right]
		right++
		if _, ok := need[c]; ok {
			window[c]++
			if window[c] == need[c] {
				valid++
			}
		}

		for valid == len(need) {
			if right-left == s1l {
				return true
			}
			cleft := s2[left]
			left++
			if _, ok := need[cleft]; ok {
				if window[cleft] == need[cleft] {
					valid--
				}
				window[cleft]--
			}
		}
	}
	return false
}

// 40. 组合总和 II
// https://leetcode.cn/problems/combination-sum-ii/
func combinationSum2(candidates []int, target int) [][]int {
	var result [][]int
	cl := len(candidates)
	pathSum := 0

	sort.Ints(candidates)
	var backtrack func(path []int, usedList []bool, start int)
	backtrack = func(path []int, usedList []bool, start int) {
		if pathSum == target {
			tmpPath := make([]int, len(path))
			copy(tmpPath, path)
			result = append(result, tmpPath)
			return
		}
		if pathSum > target {
			return
		}

		// i := start 是为了去重
		for i := start; i < cl; i++ {
			// usedList 此时是为了保证相同元素的相对位置不变
			if i > 0 && candidates[i] == candidates[i-1] && !usedList[i-1] {
				continue
			}
			path = append(path, candidates[i])
			pathSum += candidates[i]
			usedList[i] = true
			backtrack(path, usedList, i+1)
			path = path[:len(path)-1]
			pathSum -= candidates[i]
			usedList[i] = false
		}
	}

	var path []int
	usedList := make([]bool, cl)
	backtrack(path, usedList, 0)
	return result
}

// 322. 零钱兑换
// https://leetcode.cn/problems/coin-change/
func coinChange(coins []int, amount int) int {
	//dpTable := make(map[int]int)
	//for i := 0; i <= amount; i++ {
	//	dpTable[i] = -666
	//}
	//var dp func(coins []int, amount int) int
	//dp = func(coins []int, amount int) int {
	//	if amount == 0 {
	//		return 0
	//	}
	//	if amount < 0 {
	//		return -1
	//	}
	//	if dpTable[amount] != -666 {
	//		return dpTable[amount]
	//	}
	//	result := 1<<31 - 1
	//	for _, coin := range coins {
	//		sub := dp(coins, amount-coin)
	//		if sub == -1 {
	//			continue
	//		}
	//		if result > sub {
	//			result = sub
	//		}
	//	}
	//
	//	if result == 1<<31-1 {
	//		dpTable[amount] = -1
	//	} else {
	//		dpTable[amount] = result + 1
	//	}
	//	return dpTable[amount]
	//}
	//return dp(coins, amount)
	// 2023 新写,从底向上
	if amount < 0 {
		return -1
	}
	dp := make([]int, amount+1)
	for i := 1; i <= amount; i++ {
		dp[i] = amount + 1
	}
	for i := 0; i <= amount; i++ {
		for _, coin := range coins {
			if i < coin {
				continue
			}
			dp[i] = min(dp[i], dp[i-coin]+1)
		}
	}
	if dp[amount] == amount+1 {
		return -1
	}
	return dp[amount]
}

// 104. 二叉树的最大深度
// https://leetcode.cn/problems/maximum-depth-of-binary-tree/
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	leftMax := maxDepth(root.Left)
	rightMax := maxDepth(root.Right)
	if leftMax > rightMax {
		return leftMax + 1
	}
	return rightMax + 1
}

// 283. 移动零
// https://leetcode.cn/problems/move-zeroes/
func moveZeroes(nums []int) {
	slow := 0
	fast := 0
	l := len(nums)
	for fast < l {
		if nums[fast] != 0 {
			nums[slow] = nums[fast]
			slow++
		}
		fast++
	}

	for i := slow; i < l; i++ {
		nums[i] = 0
	}
}

// 86. 分隔链表
// https://leetcode.cn/problems/partition-list/
func partition(head *ListNode, x int) *ListNode {
	dummy1 := new(ListNode)
	dummy2 := new(ListNode)
	p1 := dummy1
	p2 := dummy2
	p := head

	for p != nil {
		if p.Val < x {
			p1.Next = p
			p1 = p1.Next
		} else {
			p2.Next = p
			p2 = p2.Next
		}
		temp := p.Next
		p.Next = nil
		p = temp
	}

	p1.Next = dummy2.Next
	return dummy1.Next
}

// 21. 合并两个有序链表
// https://leetcode.cn/problems/merge-two-sorted-lists/
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	dummy := new(ListNode)
	p := dummy
	for list1 != nil && list2 != nil {
		if list1.Val > list2.Val {
			p.Next = list2
			list2 = list2.Next
		} else {
			p.Next = list1
			list1 = list1.Next
		}
		p = p.Next
	}
	if list1 != nil {
		p.Next = list1
	}
	if list2 != nil {
		p.Next = list2
	}
	return dummy.Next
}

// 88. 合并两个有序数组
// https://leetcode-cn.com/problems/merge-sorted-array/submissions/
func merge_2(nums1 []int, m int, nums2 []int, n int) {
	for p := m + n - 1; m > 0 && n > 0; p-- {
		if nums2[n-1] >= nums1[m-1] {
			nums1[p] = nums2[n-1]
			n--
		} else {
			nums1[p] = nums1[m-1]
			m--
		}
	}
	for ; n > 0; n-- {
		nums1[n-1] = nums2[n-1]
	}
}

// 338. 比特位计数
// https://leetcode-cn.com/problems/counting-bits/
func countBits(num int) []int {
	highBit := 0
	res := make([]int, num+1)

	for i := 1; i <= num; i++ {
		if i&(i-1) == 0 {
			highBit = i
		}

		res[i] = res[i-highBit] + 1
	}

	return res
}

// 剑指 Offer 03. 数组中重复的数字
// https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/solution/goduo-jie-fa-lets-go-by-allentime/
func findRepeatNumber(nums []int) int {
	for i := 0; i < len(nums); i++ {
		for i != nums[i] {
			if nums[i] == nums[nums[i]] {
				return nums[i]
			} else {
				nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
			}
		}
	}

	return -1
}

// 剑指 Offer 10- II. 青蛙跳台阶问题
// https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/submissions/
func numWays(n int) int {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return 1
	}

	a, b := 1, 1
	for index := 0; index < n; index++ {
		a, b = b, (a+b)%1000000007
	}

	return a
}

// 830. 较大分组的位置
// https://leetcode-cn.com/problems/positions-of-large-groups/
func largeGroupPositions(s string) [][]int {
	var preChar uint8
	num := 0
	res := make([][]int, 0)
	temp := make([]int, 2)

	for i := 0; i < len(s); i++ {
		if s[i] != preChar {
			if temp[1] != 0 {
				res = append(res, temp)
				temp = []int{0, 0}
			}
			preChar = s[i]
			temp[0] = i
			num = 1
		} else {
			num++
		}

		if num >= 3 {
			temp[1] = i
		}
	}

	if temp[1] != 0 {
		res = append(res, temp)
		temp = []int{0, 0}
	}

	return res
}

// 509. 斐波那契数
// https://leetcode-cn.com/problems/fibonacci-number/
func fib(n int) int {
	if n == 0 {
		return 0
	}
	if n == 1 {
		return 1
	}

	a, b := 0, 1
	for index := 0; index < n; index++ {
		a, b = b, a+b
	}

	return a
}

// 1046. 最后一块石头的重量
// https://leetcode-cn.com/problems/last-stone-weight/
func lastStoneWeight(stones []int) int {
	sort.Sort(sort.Reverse(sort.IntSlice(stones)))

	ls := len(stones)
	for ; ls > 1; ls-- {
		diff := abs(stones[0] - stones[1])
		stones = append(stones[2:], diff)
		sort.Sort(sort.Reverse(sort.IntSlice(stones)))
	}

	return stones[0]
}

// 135. 分发糖果 (使用决策树)
// https://leetcode-cn.com/problems/candy/
func candy(ratings []int) int {
	lr := len(ratings)

	if lr == 0 {
		return 0
	}

	r2 := make([]int, lr)
	index := 1
	r2[0] = 1

	for index < lr {
		if r2[index] == 0 {
			r2[index]++
		}

		if index == 0 {
			index++
			continue
		}

		if ratings[index] > ratings[index-1] && r2[index] <= r2[index-1] {
			r2[index] = r2[index-1] + 1
		} else if ratings[index] < ratings[index-1] && r2[index] >= r2[index-1] {
			r2[index-1]++
			index--
			continue
		}

		index++
	}

	sum := 0
	for i := 0; i < len(r2); i++ {
		sum += r2[i]
	}

	return sum
}

// 387. 字符串中的第一个唯一字符
// https://leetcode-cn.com/problems/first-unique-character-in-a-string/
func firstUniqChar(s string) int {
	a := [26]int{}
	for i := 0; i < len(s); i++ {
		a[s[i]-'a'] = i
	}

	for j := 0; j < len(s); j++ {
		if j == a[s[j]-'a'] {
			return j
		} else {
			a[s[j]-'a'] = -1
		}
	}

	return -1
}

// 746. 使用最小花费爬楼梯
// https://leetcode-cn.com/problems/min-cost-climbing-stairs/
func minCostClimbingStairs(cost []int) int {
	lc := len(cost)
	dbtable := make([]int, lc+1)

	for i := 2; i <= lc; i++ {
		dbtable[i] = min(dbtable[i-1]+cost[i-1], dbtable[i-2]+cost[i-2])
	}

	return dbtable[lc]
}

// 416. 分割等和子集
// https://leetcode-cn.com/problems/partition-equal-subset-sum/
func canPartition(nums []int) bool {
	sum, ln := 0, len(nums)
	for i := 0; i < ln; i++ {
		sum += nums[i]
	}

	if sum%2 != 0 {
		return false
	}
	sum = sum / 2

	db1 := make([]bool, sum+1)
	db1[0] = true

	for i := 1; i <= ln; i++ {
		for j := sum; j >= 0; j-- {
			if j-nums[i-1] >= 0 {
				db1[j] = db1[j] || db1[j-nums[i-1]]
			}
		}
	}

	return db1[sum]
}

// 372. 超级次方
// https://leetcode-cn.com/problems/super-pow/
func superPow(a int, b []int) int {
	base := 1337
	var sPow func(a int, b []int) int
	var myPow func(a int, k int) int

	myPow = func(a int, k int) int {
		res := 1
		//
		//for i := 0;i<k;i++ {
		//	res *= a
		//	res %= base
		//}
		if k == 0 {
			return 1
		}

		if k%2 == 1 {
			res = (a * myPow(a, k-1)) % base
		} else {
			sub := myPow(a, k/2)
			res = (sub * sub) % base
		}

		return res % base
	}

	sPow = func(a int, b []int) int {
		if len(b) == 0 {
			return 1
		}

		last := b[len(b)-1]
		b = b[:len(b)-1]

		part1 := myPow(a, last)
		part2 := myPow(sPow(a, b), 10)

		return (part1 * part2) % base
	}

	return sPow(a, b)
}

// 172. 阶乘后的零
// https://leetcode-cn.com/problems/factorial-trailing-zeroes/
func trailingZeroes(n int) int {
	d, res := 5, 0

	for n >= d {
		res += n / d
		d *= 5
	}

	return res
}

// 231. 2的幂
// https://leetcode-cn.com/problems/power-of-two/
func isPowerOfTwo(n int) bool {
	if n > 0 && n&(n-1) == 0 {
		return true
	}
	return false
}

// 241. 为运算表达式设计优先级
// https://leetcode-cn.com/problems/different-ways-to-add-parentheses/
func diffWaysToCompute(input string) []int {
	li := len(input)

	if li == 0 {
		return []int{0}
	}

	var res []int
	for i := 0; i < li; i++ {
		if input[i] == '+' || input[i] == '-' || input[i] == '*' {
			left := diffWaysToCompute(input[0:i])
			right := diffWaysToCompute(input[i+1:])

			for j := 0; j < len(left); j++ {
				for k := 0; k < len(right); k++ {
					if input[i] == '+' {
						res = append(res, left[j]+right[k])
					} else if input[i] == '-' {
						res = append(res, left[j]-right[k])
					} else if input[i] == '*' {
						res = append(res, left[j]*right[k])
					}

				}
			}
		}
	}

	if len(res) == 0 {
		return []int{toInt(input)}
	}

	return res
}

func toInt(s string) int {
	sum := 0
	for i := 0; i < len(s); i++ {
		sum = sum*10 + int(s[i]-'0')
	}
	return sum
}

// 判断子序列
// https://leetcode-cn.com/problems/is-subsequence/
func isSubsequence(s string, t string) bool {
	ls, lt := len(s), len(t)
	m := make(map[uint8][]int)

	for i := 0; i < lt; i++ {
		m[t[i]] = append(m[t[i]], i)
	}

	k := 0
	for j := 0; j < ls; j++ {
		sChar := s[j]
		if _, ok := m[sChar]; ok {
			index := leftBound(m[sChar], k)
			if index < len(m[sChar]) {
				k = m[sChar][index] + 1
				continue
			}
		}

		return false
	}

	return true
}

// 398. 随机数索引
// https://leetcode-cn.com/problems/random-pick-index/
type Solution struct {
	nums []int
	r    *rand.Rand
}

func Constructor3(nums []int) Solution {
	return Solution{
		nums: nums,
		r:    rand.New(rand.NewSource(time.Now().Unix())),
	}
}

func (this *Solution) Pick(target int) int {
	res := -1
	step := 0
	for i := 0; i < len(this.nums); i++ {
		if this.nums[i] == target {
			step++
			if res == -1 {
				res = i
				continue
			}

			if this.r.Intn(step) == 0 {
				res = i
			}
		}
	}
	return res
}

// 448. 找到所有数组中消失的数字
// https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/
func findDisappearedNumbers(nums []int) []int {
	var res []int
	ln := len(nums)

	for i := 0; i < ln; i++ {
		index := abs(nums[i]) - 1
		if nums[index] > 0 {
			nums[index] = nums[index] * -1
		}
	}

	for i := 0; i < ln; i++ {
		if nums[i] > 0 {
			res = append(res, i+1)
		}
	}

	return res
}

// 234. 回文链表
// https://leetcode-cn.com/problems/palindrome-linked-list/
func isPalindrome(head *ListNode) bool {
	slow, fast := head, head

	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	if fast != nil {
		slow = slow.Next
	}

	res := true
	left, right := head, reverseList2(slow)
	for left != nil && right != nil {
		if left.Val != right.Val {
			res = false
			break
		}
		left = left.Next
		right = right.Next
	}
	return res
}

// 645. 错误的集合
// https://leetcode-cn.com/problems/set-mismatch/
func findErrorNums(nums []int) []int {
	dup, miss := 0, 0
	ln := len(nums)
	for i := 0; i < ln; i++ {
		ni := abs(nums[i])
		if nums[ni-1] < 0 {
			dup = ni
			continue
		}
		nums[ni-1] = -nums[ni-1]
	}

	for i := 0; i < ln; i++ {
		if nums[i] > 0 {
			miss = i + 1
			break
		}
	}

	return []int{dup, miss}
}

// 268. 丢失的数字
// https://leetcode-cn.com/problems/missing-number/
func missingNumber(nums []int) int {
	ln := len(nums)
	res := ln

	for i := 0; i < ln; i++ {
		// res += i - nums[i]
		res ^= i ^ nums[i]
	}
	return res
}

// 20. 有效的括号
// https://leetcode-cn.com/problems/valid-parentheses/
func isValid(s string) bool {
	var stack []uint8

	for i := 0; i < len(s); i++ {
		if s[i] == '(' || s[i] == '{' || s[i] == '[' {
			stack = append(stack, s[i])
		}

		ls := len(stack)
		if ls == 0 {
			return false
		}

		if s[i] == ')' {
			if stack[ls-1] != '(' {
				return false
			}
			stack = stack[:ls-1]
		}

		if s[i] == '}' {
			if stack[ls-1] != '{' {
				return false
			}
			stack = stack[:ls-1]
		}

		if s[i] == ']' {
			if stack[ls-1] != '[' {
				return false
			}
			stack = stack[:ls-1]
		}
	}

	return len(stack) == 0
}

// 5. 最长回文子串
// https://leetcode-cn.com/problems/longest-palindromic-substring/
func longestPalindrome(s string) string {
	res := ""
	ls := len(s)
	var longStr func(l int, r int) string

	longStr = func(l int, r int) string {
		for l >= 0 && r < ls && s[l] == s[r] {
			l--
			r++
		}
		return s[l+1 : r]
	}

	for i := 0; i < ls; i++ {
		s1 := longStr(i, i)
		s2 := longStr(i, i+1)

		if len(s1) > len(res) {
			res = s1
		}

		if len(s2) > len(res) {
			res = s2
		}
	}

	return res
}

// 26. 删除排序数组中的重复项
// https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/
func removeDuplicates(nums []int) int {
	ln := len(nums)
	if ln == 0 {
		return 0
	}

	i := 0
	for j := 1; j < ln; j++ {
		if nums[i] != nums[j] {
			nums[i+1] = nums[j]
			i++
		}
	}

	return i + 1
}

// 1011. 在 D 天内送达包裹的能力
// https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/
func shipWithinDays(weights []int, D int) int {
	maxSpeed, lw := 0, len(weights)
	maxWeight := 0

	for i := 0; i < lw; i++ {
		if weights[i] > maxWeight {
			maxWeight = weights[i]
		}
		maxSpeed += weights[i]
	}

	canFinish := func(speed int) bool {
		t := 1
		tmp := 0

		for j := 0; j < lw; j++ {
			tmp += weights[j]

			if tmp > speed {
				t++
				tmp = weights[j]
			}
		}

		return t <= D
	}

	left, right := maxWeight, maxSpeed
	for left <= right {
		mid := left + (right-left)/2
		if canFinish(mid) {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	return left
}

// 875. 爱吃香蕉的珂珂
// https://leetcode-cn.com/problems/koko-eating-bananas/
func minEatingSpeed(piles []int, H int) int {
	maxSpeed, lp := 0, len(piles)

	for i := 0; i < lp; i++ {
		if piles[i] > maxSpeed {
			maxSpeed = piles[i]
		}
	}

	canFinish := func(speed int) bool {
		t := 0
		for j := 0; j < lp; j++ {
			t += piles[j] / speed
			if piles[j]%speed > 0 {
				t += 1
			}
		}
		return t <= H
	}

	left, right := 1, maxSpeed
	for left <= right {
		mid := left + (right-left)/2
		if canFinish(mid) {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	return left
}

// 计数质数
// https://leetcode-cn.com/problems/count-primes/
func countPrimes(n int) int {
	isNotPrime := make([]bool, n+1)
	count := 0

	for i := 2; i*i < n; i++ {
		if isNotPrime[i] == false {
			for j := i * i; j < n; j += i {
				isNotPrime[j] = true
			}
		}
	}

	for i := 2; i < n; i++ {
		if isNotPrime[i] == false {
			count++
		}
	}

	return count
}

// LRU缓存机制
// https://leetcode-cn.com/problems/lru-cache/
type DoubleNode struct {
	Key   int
	Value int
	Next  *DoubleNode
	Pre   *DoubleNode
}

type DoubleList struct {
	Head *DoubleNode
	Tail *DoubleNode
	Len  int
}

func DList() *DoubleList {
	dl := &DoubleList{}
	dl.Head = &DoubleNode{Key: 0, Value: 0}
	dl.Tail = &DoubleNode{Key: 0, Value: 0}
	dl.Head.Next = dl.Tail
	dl.Tail.Pre = dl.Head
	return dl
}

func (dl *DoubleList) Append(d *DoubleNode) {
	d.Next = dl.Tail
	d.Pre = dl.Tail.Pre

	dl.Tail.Pre.Next = d
	dl.Tail.Pre = d
	dl.Len++
}

func (dl *DoubleList) Delete(d *DoubleNode) {
	d.Pre.Next = d.Next
	d.Next.Pre = d.Pre
	dl.Len--
}

func (dl *DoubleList) DeleteHead() *DoubleNode {
	if dl.Head.Next == dl.Tail {
		return nil
	}

	tmp := dl.Head.Next
	dl.Delete(tmp)
	return tmp
}

type LRUCache struct {
	Cap  int
	Len  int
	Map  map[int]*DoubleNode
	List *DoubleList
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{Cap: capacity}
	l.Map = make(map[int]*DoubleNode, capacity)
	l.List = DList()
	return l
}

func (this *LRUCache) Get(key int) int {
	if value, ok := this.Map[key]; ok {
		this.List.Delete(value)
		this.List.Append(value)

		return value.Value
	}

	return -1
}

func (this *LRUCache) Put(key int, value int) {
	// if v, ok := this.Map[key]; ok {
	// 	this.List.Delete(v)

	// 	d := &DoubleNode{Key: key, Value: value}
	// 	this.List.Append(d)
	// 	this.Map[key] = d
	// 	return
	// }

	// if this.Len == this.Cap {
	// 	head := this.List.DeleteHead()
	// 	delete(this.Map, head.Key)
	// 	this.Len--
	// }

	// d := &DoubleNode{Key: key, Value: value}
	// this.Map[key] = d
	// this.List.Append(d)
	// this.Len++
}

// 求根到叶子节点数字之和
// https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/
func sumNumbers(root *TreeNode) int {
	sum := 0
	var path func(r *TreeNode, path int)

	path = func(r *TreeNode, pathSum int) {
		if r.Left == nil && r.Right == nil {
			sum += pathSum*10 + r.Val
		}

		if r.Left != nil {
			path(r.Left, pathSum*10+r.Val)
		}

		if r.Right != nil {
			path(r.Right, pathSum*10+r.Val)
		}
	}

	if root == nil {
		return 0
	}

	path(root, 0)

	return sum
}

// 接雨水
// https://leetcode-cn.com/problems/trapping-rain-water/
func trap(height []int) int {
	left, right := 0, len(height)-1
	lMax, rMax := 0, 0
	res := 0

	for left < right {
		if height[left] <= height[right] {
			if height[left] < lMax {
				res += lMax - height[left]
			} else {
				lMax = height[left]
			}
			left++
		} else if height[left] > height[right] {
			if height[right] < rMax {
				res += rMax - height[right]
			} else {
				rMax = height[right]
			}
			right--
		}
	}

	return res
}

func trap2(height []int) int {
	leftHigh := make([]int, len(height))
	rightHigh := make([]int, len(height))

	leftMax := height[0]
	for i := 0; i < len(height); i++ {
		if height[i] > leftMax {
			leftMax = height[i]
		}
		leftHigh[i] = leftMax
	}

	rightMax := height[len(height)-1]
	for i := len(height) - 1; i >= 0; i-- {
		if height[i] > rightMax {
			rightMax = height[i]
		}
		rightHigh[i] = rightMax
	}

	result := 0
	for i := 0; i < len(height); i++ {
		result += min(leftHigh[i], rightHigh[i]) - height[i]
	}
	return result
}

// 颜色填充
// https://leetcode-cn.com/problems/color-fill-lcci/
func floodFill(image [][]int, sr int, sc int, newColor int) [][]int {
	var fill func(sr int, sc int, oldColor int)

	if !inArea(image, sr, sc) {
		return image
	}
	oldColor := image[sr][sc]

	fill = func(sr int, sc int, oldColor int) {
		if !inArea(image, sr, sc) {
			return
		}
		if image[sr][sc] != oldColor {
			return
		}
		if image[sr][sc] == -1 {
			return
		}

		image[sr][sc] = -1

		fill(sr+1, sc, oldColor)
		fill(sr-1, sc, oldColor)
		fill(sr, sc+1, oldColor)
		fill(sr, sc-1, oldColor)

		image[sr][sc] = newColor
	}

	fill(sr, sc, oldColor)

	return image
}

func inArea(image [][]int, sr int, sc int) bool {
	return sr >= 0 && sr <= len(image)-1 && sc >= 0 && sc <= len(image[sr])-1
}

// 字符串相乘
// https://leetcode-cn.com/problems/multiply-strings/
func multiply(num1 string, num2 string) string {
	l1, l2, l3 := len(num1), len(num2), len(num1)+len(num2)
	res := make([]uint8, l3)

	for i := l2 - 1; i >= 0; i-- {
		for j := l1 - 1; j >= 0; j-- {
			high, low := i+j, i+j+1
			sum := res[low] + (num2[i]-'0')*(num1[j]-'0')
			res[low] = sum % 10
			res[high] += sum / 10
		}
	}

	start := -1
	for i := 0; i < len(res); i++ {
		if res[i] != 0 && start == -1 {
			start = i
		}

		res[i] = res[i] + '0'
	}

	if start == -1 {
		return "0"
	} else {
		return string(res[start:])
	}
}

// 和为K的子数组
// https://leetcode-cn.com/problems/subarray-sum-equals-k/
func subarraySum(nums []int, k int) int {
	ans, sumI := 0, 0
	preSum := map[int]int{}
	preSum[0] = 1

	for i := 0; i < len(nums); i++ {
		sumI += nums[i]
		if _, ok := preSum[sumI-k]; ok {
			ans += preSum[sumI-k]
		}
		preSum[sumI] += 1
	}

	return ans
}

// 计算器
// https://leetcode-cn.com/problems/calculator-lcci/
func calculate(s string) int {

	var cal func() int
	cal = func() int {
		stack := []int{}
		sign := uint8('+')
		num := 0

		for len(s) > 0 {
			c := s[0]

			if c == '(' {
				s = s[1:]
				num = cal()
			}

			for c >= '0' && c <= '9' {
				num = 10*num + int(c-'0')
				s = s[1:]
				if len(s) == 0 {
					break
				}
				c = s[0]
			}

			if len(s) > 0 {
				s = s[1:]
			}

			if (c < '0' || c > '9') && c != ' ' || len(s) == 0 {
				switch sign {
				case '+':
					stack = append(stack, num)
					break
				case '-':
					stack = append(stack, -num)
					break
				case '*':
					top := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					stack = append(stack, top*num)
					break
				case '/':
					top := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					stack = append(stack, top/num)
					break
				}
				sign = c
				num = 0
			}

			if c == ')' {
				break
			}
		}

		sumC := 0
		for i := 0; i < len(stack); i++ {
			sumC += stack[i]
		}

		return sumC
	}

	return cal()
}

// 位1的个数
// https://leetcode-cn.com/problems/number-of-1-bits/
func hammingWeight(num uint32) int {
	res := 0

	for num != 0 {
		res++
		num = num & (num - 1)
	}

	return res
}

// 最长不含重复字符的子字符串
// https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/
func lengthOfLongestSubstring(s string) int {
	window := make(map[uint8]int)
	left, right, res := 0, 0, 0
	for right < len(s) {
		c := s[right]
		right++
		window[c]++
		for window[c] > 1 {
			cleft := s[left]
			left++
			window[cleft]--
		}
		if right-left > res {
			res = right - left
		}
	}
	return res
}

//找到字符串中所有字母异位词
//https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/

func findAnagrams(s2 string, s1 string) []int {
	var res []int
	valid := 0
	left := 0
	right := 0
	s1l := len(s1)
	need := make(map[uint8]int)
	window := make(map[uint8]int)
	for i := 0; i < s1l; i++ {
		need[s1[i]]++
	}

	for right < len(s2) {
		c := s2[right]
		right++
		if _, ok := need[c]; ok {
			window[c]++
			if window[c] == need[c] {
				valid++
			}
		}

		for valid == len(need) {
			if right-left == s1l {
				res = append(res, left)
			}
			cleft := s2[left]
			left++
			if _, ok := need[cleft]; ok {
				if window[cleft] == need[cleft] {
					valid--
				}
				window[cleft]--
			}
		}
	}
	return res
}

// 76. 最小覆盖子串
// https://leetcode-cn.com/problems/minimum-window-substring/
func minWindow(s string, t string) string {
	lt, ls := len(t), len(s)
	left, right, rleft, rright := 0, 0, 0, 1<<32

	match := 0
	nMap := make(map[uint8]int, lt)
	wMap := make(map[uint8]int, lt)

	for i := 0; i < lt; i++ {
		nMap[t[i]]++
	}

	for right < ls {
		if _, ok := nMap[s[right]]; ok {
			wMap[s[right]]++
			if wMap[s[right]] == nMap[s[right]] {
				match++
			}
		}

		for match == len(nMap) && match > 0 {
			if right-left < rright-rleft {
				rright, rleft = right, left
			}

			if _, ok := nMap[s[left]]; ok {
				wMap[s[left]]--
				if wMap[s[left]] < nMap[s[left]] {
					match--
				}
			}
			left++
		}
		right++
	}

	if rright != 1<<32 {
		return s[rleft : rright+1]
	}

	return ""
}

// 三数之和
// https://leetcode-cn.com/problems/3sum/
func threeSum(nums []int) [][]int {
	var res [][]int

	sort.Slice(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	for first := 0; first <= len(nums)-1; {
		vFirst := -nums[first]

		second, third := first+1, len(nums)-1
		for second < third {
			if nums[second]+nums[third] == vFirst {
				res = append(res, []int{nums[first], nums[second], nums[third]})

				vThird := nums[third]
				for third > second && nums[third] == vThird {
					third--
				}

				vSecond := nums[second]
				for second < third && nums[third] == vSecond {
					second++
				}
			}

			if nums[second]+nums[third] > vFirst {
				vThird := nums[third]
				for third > second && nums[third] == vThird {
					third--
				}
			}

			if nums[second]+nums[third] < vFirst {
				vSecond := nums[second]
				for second < third && nums[second] == vSecond {
					second++
				}
			}
		}

		for first < len(nums) && nums[first] == -vFirst {
			first++
		}
	}

	return res
}

// 三数之和
// https://leetcode-cn.com/problems/3sum/
func threeSum1(nums []int) [][]int {
	var res [][]int

	sort.Slice(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})

	for i := 0; i <= len(nums)-1; {
		vi := nums[i]

		twoRes := twoNum(nums[i+1:], -vi)

		for j := 0; j < len(twoRes); j++ {
			res = append(res, []int{vi, twoRes[j][0], twoRes[j][1]})
		}

		for {
			i++
			if i >= len(nums) || vi != nums[i] {
				break
			}
		}
	}

	return res
}

// 输入有序数组,求和等于target的不重复值
func twoNum(numbers []int, target int) [][]int {
	l, r := 0, len(numbers)-1
	var res [][]int

	for l < r {
		if numbers[l]+numbers[r] == target {
			res = append(res, []int{numbers[l], numbers[r]})

			vr := numbers[r]
			for r > 0 && numbers[r] == vr {
				r--
			}

			vl := numbers[l]
			for l < len(numbers)-1 && numbers[l] == vl {
				l++
			}
		} else if numbers[l]+numbers[r] > target {
			vr := numbers[r]
			for r > 0 && numbers[r] == vr {
				r--
			}
		} else if numbers[l]+numbers[r] < target {
			vl := numbers[l]
			for l < len(numbers)-1 && numbers[l] == vl {
				l++
			}
		}
	}

	return res
}

// 两数之和 II - 输入有序数组
// https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted/
func twoSum2(numbers []int, target int) []int {
	l, r := 0, len(numbers)-1

	for l < r {
		if numbers[l]+numbers[r] == target {
			return []int{l + 1, r + 1}
		} else if numbers[l]+numbers[r] > target {
			r--
		} else if numbers[l]+numbers[r] < target {
			l++
		}
	}

	return []int{}
}

// 两数之和
// https://leetcode-cn.com/problems/two-sum/
func twoSum1(nums []int, target int) []int {
	nl := len(nums)
	if nl < 2 {
		return []int{}
	}

	tmp := make(map[int]int, nl)

	for i := 0; i < nl; i++ {
		t := target - nums[i]
		if tt, ok := tmp[t]; ok {
			return []int{i, tt}
		} else {
			tmp[nums[i]] = i
		}
	}

	return []int{}
}

// 组合
// https://leetcode-cn.com/problems/combinations/
func combine(n int, k int) [][]int {
	var res [][]int
	nl := n
	if nl == 0 {
		return res
	}

	var backtrack func(path []int, start int)
	backtrack = func(path []int, start int) {
		if len(path) == k {
			tmp := make([]int, k)
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for i := start + 1; i < nl; i++ {
			path = append(path, i+1)
			backtrack(path, i)
			path = path[:len(path)-1]
		}
	}

	backtrack(make([]int, 0), -1)

	return res
}

// 子集 II
// https://leetcode-cn.com/problems/subsets-ii/
func subsetsWithDup(nums []int) [][]int {
	res := make([][]int, 0)
	nl := len(nums)
	if nl == 0 {
		return res
	}

	used := make([]bool, nl)
	sort.Ints(nums)

	var backtrack func(path []int, start int, used []bool)
	backtrack = func(path []int, start int, used []bool) {
		tmp := make([]int, len(path))
		copy(tmp, path)
		res = append(res, tmp)
		for i := start + 1; i < nl; i++ {
			if i > 0 && nums[i] == nums[i-1] && !used[i-1] {
				continue
			}
			tmp = append(tmp, nums[i])
			used[i] = true
			backtrack(tmp, i, used)
			tmp = tmp[:len(tmp)-1]
			used[i] = false
		}
	}

	backtrack([]int{}, -1, used)
	return res
}

// 子集
// https://leetcode-cn.com/problems/subsets/
func subsets(nums []int) [][]int {
	res := make([][]int, 0)
	nl := len(nums)
	if nl == 0 {
		return res
	}
	var backtrack func(path []int, start int)
	backtrack = func(path []int, start int) {
		tmp := make([]int, len(path))
		copy(tmp, path)
		res = append(res, tmp)
		for i := start + 1; i < nl; i++ {
			tmp = append(tmp, nums[i])
			backtrack(tmp, i)
			tmp = tmp[:len(tmp)-1]
		}
	}
	backtrack([]int{}, -1)
	return res
}

// N皇后 II
// https://leetcode-cn.com/problems/n-queens-ii/
func totalNQueens2(n int) int {
	nl := n
	var res [][]int
	if nl == 0 {
		return 0
	}

	isValid := func(path []int, row int, col int) bool {
		for i := 0; i <= row-1; i++ {
			if path[i] == 1<<col {
				return false
			}
		}

		for i, j := row-1, col-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
			if path[i] == 1<<j {
				return false
			}
		}

		for i, j := row-1, col+1; i >= 0 && j >= 0 && j < nl; i, j = i-1, j+1 {
			if path[i] == 1<<j {
				return false
			}
		}
		return true
	}

	makeQ := func(totalLen int, pos int) int {
		if pos < 0 {
			return 0
		}
		return 1 << pos
	}

	path := make([]int, nl)

	var backtrack func(row int)
	backtrack = func(row int) {
		if row == nl {
			tmp := make([]int, nl)
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for i := 0; i < nl; i++ {
			if isValid(path, row, i) {
				path[row] = makeQ(nl, i)
				backtrack(row + 1)
				path[row] = makeQ(nl, -1)
			}
		}
	}

	backtrack(0)
	return len(res)
}

// N 皇后
// https://leetcode-cn.com/problems/n-queens/
func solveNQueens(n int) [][]string {
	nl := n
	var res [][]string
	if nl == 0 {
		return res
	}

	isValid := func(path []string, row int, col int) bool {
		for i := 0; i <= row-1; i++ {
			if path[i][col] == 'Q' {
				return false
			}
		}

		for i, j := row-1, col-1; i >= 0 && j >= 0; i, j = i-1, j-1 {
			if path[i][j] == 'Q' {
				return false
			}
		}

		for i, j := row-1, col+1; i >= 0 && j >= 0 && j < nl; i, j = i-1, j+1 {
			if path[i][j] == 'Q' {
				return false
			}
		}
		return true
	}
	makeQ := func(totalLen int, pos int) string {
		if pos < 0 {
			return strings.Repeat(".", totalLen)
		}
		return strings.Repeat(".", pos) + "Q" + strings.Repeat(".", totalLen-pos-1)
	}

	path := make([]string, nl)
	for i := 0; i < nl; i++ {
		path[i] = makeQ(nl, -1)
	}

	var backtrack func(row int)
	backtrack = func(row int) {
		if row == nl {
			tmp := make([]string, nl)
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for i := 0; i < nl; i++ {
			if isValid(path, row, i) {
				path[row] = makeQ(nl, i)
				backtrack(row + 1)
				path[row] = makeQ(nl, 0)
			}
		}
	}

	backtrack(0)
	return res
}

// 全排列 II
// https://leetcode-cn.com/problems/permutations-ii/
func permuteUnique(nums []int) [][]int {
	var res [][]int
	nl := len(nums)
	if nl == 0 {
		return res
	}

	useds := make([]bool, nl)
	sort.Ints(nums)
	var backtrack func(path []int, used []bool)
	backtrack = func(path []int, used []bool) {
		if len(path) == nl {
			tmp := make([]int, len(nums))
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for i := 0; i < nl; i++ {
			if i > 0 && nums[i] == nums[i-1] && !useds[i-1] {
				continue
			}

			if !used[i] {
				path = append(path, nums[i])
				used[i] = true

				backtrack(path, used)

				path = path[:len(path)-1]
				used[i] = false
			}
		}
	}

	backtrack(make([]int, 0), useds)

	return res
}

// 全排列
// https://leetcode-cn.com/problems/permutations/
func permute(nums []int) [][]int {
	var res [][]int
	nl := len(nums)
	if nl == 0 {
		return res
	}

	useds := make([]bool, nl)

	var backtrack func(path []int, used []bool)
	backtrack = func(path []int, used []bool) {
		if len(path) == nl {
			tmp := make([]int, len(nums))
			copy(tmp, path)
			res = append(res, tmp)
			return
		}

		for i := 0; i < nl; i++ {
			if !used[i] {
				path = append(path, nums[i])
				used[i] = true

				backtrack(path, used)

				path = path[:len(path)-1]
				used[i] = false
			}
		}
	}

	backtrack(make([]int, 0), useds)

	return res
}

// 25. K 个一组翻转链表
// https://leetcode-cn.com/problems/reverse-nodes-in-k-group/
func reverseKGroup(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}

	b := head
	for i := 0; i < k; i++ {
		if b == nil {
			return head
		}
		b = b.Next
	}

	newhead := rList(head, b)
	head.Next = reverseKGroup(b, k)

	return newhead
}

func rList(head *ListNode, b *ListNode) *ListNode {
	if head == nil {
		return head
	}

	var pre *ListNode
	now, next := head, head

	for now != b {
		next = now.Next
		now.Next = pre

		pre = now
		now = next
	}

	return pre
}

// 反转链表 II
// https://leetcode-cn.com/problems/reverse-linked-list-ii/
func reverseBetween(head *ListNode, m int, n int) *ListNode {
	if head == nil {
		return head
	}

	var rlN func(h *ListNode, n int) *ListNode
	var lastNext *ListNode
	rlN = func(h *ListNode, n int) *ListNode {
		if n == 1 {
			lastNext = h.Next
			return h
		}

		last := rlN(h.Next, n-1)
		h.Next.Next = h
		h.Next = lastNext
		return last
	}

	if m == 1 {
		return rlN(head, n)
	}

	head.Next = reverseBetween(head.Next, m-1, n-1)

	return head
}

// 反转链表
// https://leetcode-cn.com/problems/reverse-linked-list/
func reverseList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}

	var rl func(h *ListNode) *ListNode

	rl = func(h *ListNode) *ListNode {
		if h.Next == nil {
			return h
		}
		last := rl(h.Next)
		h.Next.Next = h
		h.Next = nil

		return last
	}

	return rl(head)
}

func reverseList2(head *ListNode) *ListNode {
	if head == nil {
		return head
	}

	var pre *ListNode
	now, next := head, head

	for now != nil {
		next = now.Next
		now.Next = pre

		pre = now
		now = next
	}

	return pre
}

// 滑动窗口最大值
// https://leetcode-cn.com/problems/sliding-window-maximum/
func maxSlidingWindow(nums []int, k int) []int {
	dq := &DQueue{}
	res := make([]int, 0)

	for i := 0; i < len(nums); i++ {
		if i < k-1 {
			dq.Push(nums[i])
		} else {
			dq.Push(nums[i])
			res = append(res, dq.Front())
			dq.Pop(nums[i-k+1])
		}
	}

	return res
}

// 链表中的下一个更大节点
// https://leetcode-cn.com/problems/next-greater-node-in-linked-list/
func nextLargerNodes(head *ListNode) []int {
	ans := make([]int, 0)
	stack := make([]int, 0)

	if head == nil {
		return ans
	}

	var deep func(h *ListNode)
	deep = func(h *ListNode) {
		if h.Next != nil {
			deep(h.Next)
		}

		for len(stack) > 0 && stack[len(stack)-1] <= h.Val {
			stack = stack[:len(stack)-1]
		}

		if len(stack) == 0 {
			ans = append(ans, 0)
		} else {
			ans = append(ans, stack[len(stack)-1])
		}

		stack = append(stack, h.Val)
	}

	deep(head)

	for i, j := 0, len(ans)-1; i < j; i, j = i+1, j-1 {
		ans[i], ans[j] = ans[j], ans[i]
	}

	return ans
}

// 最大二叉树 II
// https://leetcode-cn.com/problems/maximum-binary-tree-ii/
func insertIntoMaxTree(root *TreeNode, val int) *TreeNode {
	if root == nil {
		root = &TreeNode{Val: val}
		return root
	}

	temp := root
	for {
		if temp.Val < val {
			tNew := &TreeNode{Val: val, Left: temp}
			root = tNew
			return root
		} else if temp.Right == nil {
			temp.Right = &TreeNode{Val: val}
			return root
		} else if temp.Right.Val >= val {
			temp = temp.Right
		} else if temp.Right.Val < val {
			temp.Right = &TreeNode{Val: val, Left: temp.Right}
			return root
		}
	}
}

// 最大二叉树
// https://leetcode-cn.com/problems/maximum-binary-tree/
func constructMaximumBinaryTree(nums []int) *TreeNode {
	root := new(TreeNode)
	nl := len(nums)
	if nl == 0 {
		return root
	}
	root.Val = nums[0]

	for i := 1; i < nl; i++ {
		root = insertIntoMaxTree(root, nums[i])
	}

	return root
}

// 正则表达式匹配
// https://leetcode-cn.com/problems/regular-expression-matching/
func isMatch2(s string, p string) bool {
	pl, sl := len(p), len(s)
	if pl == 0 {
		if sl == 0 {
			return true
		} else {
			return false
		}
	}
	firstMatch := false
	if sl != 0 {
		firstMatch = p[0] == s[0] || p[0] == '.'
	}

	if pl >= 2 && p[1] == '*' {
		return isMatch(s, p[2:]) || (firstMatch && isMatch(s[1:], p))
	}

	return firstMatch && isMatch(s[1:], p[1:])
}

func isMatch(s string, p string) bool {
	dp := make([][]int, len(s)+1)
	for k, _ := range dp {
		dp[k] = make([]int, len(p)+1)
	}

	var isM func(si int, pj int) bool

	isM = func(si int, pj int) bool {
		if dp[si][pj] != 0 {
			return dp[si][pj] == 2
		}

		sl, pl := len(s[si:]), len(p[pj:])

		if pl == 0 {
			if sl == 0 {
				return true
			} else {
				return false
			}
		}

		firstMatch := false
		if sl != 0 {
			firstMatch = p[pj+0] == s[si+0] || p[pj+0] == '.'
		}

		if pl >= 2 && p[pj+1] == '*' {
			r := isM(si, pj+2) || (firstMatch && isM(si+1, pj))
			if r == true {
				dp[si][pj] = 2
			} else {
				dp[si][pj] = 1
			}
			return r
		}

		r := firstMatch && isM(si+1, pj+1)
		if r == true {
			dp[si][pj] = 2
		} else {
			dp[si][pj] = 1
		}
		return r
	}

	return isM(0, 0)
}

// 打家劫舍 III
// https://leetcode-cn.com/problems/house-robber-iii/
func rob3(root *TreeNode) int {
	var res []int
	var r func(node *TreeNode) []int
	r = func(node *TreeNode) []int {
		if node == nil {
			return []int{0, 0}
		}
		l, r := r(node.Left), r(node.Right)
		selected := node.Val + l[1] + r[1]
		notSelected := max(l[0], l[1]) + max(r[0], r[1])
		return []int{selected, notSelected}
	}

	res = r(root)

	return max(res[0], res[1])
}

// 打家劫舍 II
// https://leetcode-cn.com/problems/house-robber-ii/
func rob2(nums []int) int {
	nl := len(nums)

	if nl == 1 {
		return nums[0]
	}

	var r func(start int, end int) int

	r = func(start int, end int) int {
		pre1 := 0
		pre2 := 0
		for i := end; i >= start; i-- {
			temp := pre1
			pre1 = max(pre1, pre2+nums[i])
			pre2 = temp
		}

		return pre1
	}

	return max(r(0, nl-2), r(1, nl-1))
}

// 打家劫舍
// https://leetcode-cn.com/problems/house-robber/
func rob1(nums []int) int {
	nl := len(nums)

	pre1 := 0
	pre2 := 0
	for i := nl - 1; i >= 0; i-- {
		temp := pre1
		pre1 = max(pre1, pre2+nums[i])
		pre2 = temp
	}

	return pre1
}

// 买买卖股票的最佳时机 IV
// https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/
func maxProfit4(k int, prices []int) int {
	maxK := k
	pl := len(prices)
	dp := make([][][2]int, pl)

	if pl == 0 {
		return 0
	}

	if k > pl/2 {
		return maxProfit2(prices)
	}

	for i := 0; i < pl; i++ {
		for j := 0; j <= maxK; j++ {
			dp[i] = append(dp[i], [2]int{})
		}
	}

	for i := 0; i < pl; i++ {
		for j := 1; j <= maxK; j++ {
			if i == 0 {
				dp[i][j][0] = 0
				dp[i][j][1] = -prices[i]
				continue
			}

			dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])
			dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])
		}
	}

	return dp[pl-1][maxK][0]
}

// 买卖股票的最佳时机 III
// https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/
func maxProfit3(prices []int) int {
	maxK := 2
	pl := len(prices)
	dp := make([][3][2]int, pl)

	for i := 0; i < pl; i++ {
		dp[i] = [3][2]int{}
	}

	for i := 0; i < pl; i++ {
		for j := 1; j <= maxK; j++ {
			if i == 0 {
				dp[i][j][0] = 0
				dp[i][j][1] = -prices[i]
				continue
			}

			dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])
			dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])
		}
	}

	return dp[pl-1][maxK][0]
}

// 买卖股票的最佳时机 II
// https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
func maxProfit2(prices []int) int {
	pre_has := math.MinInt32
	pre_no_has := 0

	for i := 0; i < len(prices); i++ {
		temp := pre_has
		pre_has = max(pre_has, pre_no_has-prices[i])
		pre_no_has = max(pre_no_has, temp+prices[i])
	}

	return pre_no_has
}

// 买卖股票的最佳时机
// https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
func maxProfit1(prices []int) int {
	pre_has := math.MinInt32
	pre_no_has := 0

	for i := 0; i < len(prices); i++ {
		pre_no_has = max(pre_no_has, pre_has+prices[i])
		pre_has = max(pre_has, -prices[i])
	}

	return pre_no_has
}

// 删除被覆盖区间
// https://leetcode-cn.com/problems/remove-covered-intervals/
func removeCoveredIntervals(intervals [][]int) int {

	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] {
			return true
		} else if intervals[i][0] == intervals[j][0] {
			return intervals[i][1] > intervals[j][1]
		}
		return false
	})

	li := len(intervals)
	res := 0

	if li == 0 {
		return res
	}

	left := intervals[0][0]
	right := intervals[0][1]

	for i := 1; i < li; i++ {
		if left <= intervals[i][0] && intervals[i][1] <= right {
			res++
		} else if intervals[i][1] > right {
			right = intervals[i][1]
		} else if right < intervals[i][0] {
			left = intervals[i][0]
			right = intervals[i][1]
		}
	}

	return li - res
}

// 区间列表的交集
// https://leetcode-cn.com/problems/interval-list-intersections/
func intervalIntersection(A [][]int, B [][]int) [][]int {
	i, j, la, lb := 0, 0, len(A), len(B)
	res := [][]int{}
	for i < la && j < lb {
		aa := A[i]
		bb := B[j]

		if aa[1] >= bb[0] && bb[1] >= aa[0] {
			res = append(res, []int{max(aa[0], bb[0]), min(aa[1], bb[1])})
		}

		if bb[1] > aa[1] {
			i++
		} else {
			j++
		}
	}

	return res
}

// 合并区间
// https://leetcode-cn.com/problems/merge-intervals/
func merge(intervals [][]int) [][]int {
	var res [][]int

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})

	for i := 0; i < len(intervals); i++ {
		if len(res) == 0 || intervals[i][0] > res[len(res)-1][1] {
			res = append(res, intervals[i])
		} else if intervals[i][1] > res[len(res)-1][1] {
			res[len(res)-1][1] = intervals[i][1]
		}
	}

	return res
}

// 无重叠区间
// https://leetcode-cn.com/problems/non-overlapping-intervals/
func eraseOverlapIntervals(intervals [][]int) int {
	li := len(intervals)
	res := 0
	if li == 0 {
		return res
	}

	//冒泡排序
	//for i:=0;i<li;i++{
	//	for j:=0;j<li-i-1;j++{
	//		if intervals[j][1] > intervals[j+1][1] {
	//			intervals[j],intervals[j+1] = intervals[j+1],intervals[j]
	//		}
	//	}
	//}
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][1] < intervals[j][1]
	})

	end := intervals[0][1]
	for i := 1; i < li; i++ {
		if intervals[i][0] < end {
			res++
		} else {
			end = intervals[i][1]
		}
	}

	return res
}

// N叉树的前序遍历
// https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal/submissions/
func preorder(root *Node) []int {
	res := make([]int, 0)

	var front func(root *Node)
	front = func(root *Node) {
		if root != nil {
			res = append(res, root.Val)
			for _, v := range root.Children {
				front(v)
			}
		}
	}

	front(root)

	return res
}

// 特定深度节点链表
// https://leetcode-cn.com/problems/list-of-depth-lcci/submissions/
func listOfDepth(tree *TreeNode) []*ListNode {
	treeNodes := make([]*TreeNode, 0)
	listNodes := make([]*ListNode, 0)

	if tree == nil {
		return listNodes
	}

	treeNodes = append(treeNodes, tree)

	for len(treeNodes) > 0 {
		nextTreeNodes := make([]*TreeNode, 0)
		tmpNode := &ListNode{}
		headNode := tmpNode

		for i := 0; i < len(treeNodes); i++ {
			tmpNode.Next = &ListNode{Val: treeNodes[i].Val}
			tmpNode = tmpNode.Next

			if treeNodes[i].Left != nil {
				nextTreeNodes = append(nextTreeNodes, treeNodes[i].Left)
			}
			if treeNodes[i].Right != nil {
				nextTreeNodes = append(nextTreeNodes, treeNodes[i].Right)
			}
		}

		listNodes = append(listNodes, headNode.Next)
		treeNodes = nextTreeNodes
	}

	return listNodes
}
