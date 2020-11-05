package main

import (
	"math"
	"sort"
	"strings"
)

//计数质数
//https://leetcode-cn.com/problems/count-primes/
func countPrimes(n int) int {
	isNotPrime := make([]bool,n+1)
	count := 0

	for i:=2; i * i < n; i++ {
		if isNotPrime[i] == false {
			for j:= i * i; j < n; j+=i {
				isNotPrime[j] = true
			}
		}
	}

	for i:=2;i<n;i++ {
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
	if v, ok := this.Map[key]; ok {
		this.List.Delete(v)

		d := &DoubleNode{Key: key, Value: value}
		this.List.Append(d)
		this.Map[key] = d
		return
	}

	if this.Len == this.Cap {
		head := this.List.DeleteHead()
		delete(this.Map, head.Key)
		this.Len--
	}

	d := &DoubleNode{Key: key, Value: value}
	this.Map[key] = d
	this.List.Append(d)
	this.Len++
}

//求根到叶子节点数字之和
//https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/
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

//接雨水
//https://leetcode-cn.com/problems/trapping-rain-water/
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

//颜色填充
//https://leetcode-cn.com/problems/color-fill-lcci/
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

//字符串相乘
//https://leetcode-cn.com/problems/multiply-strings/
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

//和为K的子数组
//https://leetcode-cn.com/problems/subarray-sum-equals-k/
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

//计算器
//https://leetcode-cn.com/problems/calculator-lcci/
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

//位1的个数
//https://leetcode-cn.com/problems/number-of-1-bits/
func hammingWeight(num uint32) int {
	res := 0

	for num != 0 {
		res++
		num = num & (num - 1)
	}

	return res
}

//最长不含重复字符的子字符串
//https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/
func lengthOfLongestSubstring(s string) int {
	left, right, ls, rl, rr := 0, 0, len(s), 0, -1
	nMap := map[uint8]int{}

	for right < ls {
		nMap[s[right]]++

		for nMap[s[right]] > 1 {
			if _, ok := nMap[s[left]]; ok {
				nMap[s[left]]--
			}
			left++
		}

		if right-left > rr-rl {
			rr, rl = right, left
		}

		right++
	}

	if rr != -1 {
		return rr - rl + 1
	}

	return 0
}

// 找到字符串中所有字母异位词
//https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/
func findAnagrams(s string, p string) []int {
	left, right := 0, 0
	var res []int
	t := p
	lt, ls := len(t), len(s)

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
			if right-left+1 == lt {
				res = append(res, left)
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

	return res
}

//最小覆盖子串
//https://leetcode-cn.com/problems/minimum-window-substring/
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

//三数之和
//https://leetcode-cn.com/problems/3sum/
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

//三数之和
//https://leetcode-cn.com/problems/3sum/
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

//输入有序数组,求和等于target的不重复值
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

//两数之和
//https://leetcode-cn.com/problems/two-sum/
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

//组合
//https://leetcode-cn.com/problems/combinations/
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

//子集 II
//https://leetcode-cn.com/problems/subsets-ii/
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

//子集
//https://leetcode-cn.com/problems/subsets/
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

//N皇后 II
//https://leetcode-cn.com/problems/n-queens-ii/
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

//N 皇后
//https://leetcode-cn.com/problems/n-queens/
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
//https://leetcode-cn.com/problems/permutations-ii/
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

//全排列
//https://leetcode-cn.com/problems/permutations/
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

//反转链表 II
//https://leetcode-cn.com/problems/reverse-linked-list-ii/
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

//反转链表
//https://leetcode-cn.com/problems/reverse-linked-list/
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

//滑动窗口最大值
//https://leetcode-cn.com/problems/sliding-window-maximum/
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

//链表中的下一个更大节点
//https://leetcode-cn.com/problems/next-greater-node-in-linked-list/
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

//最大二叉树 II
//https://leetcode-cn.com/problems/maximum-binary-tree-ii/
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

//最大二叉树
//https://leetcode-cn.com/problems/maximum-binary-tree/
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

//正则表达式匹配
//https://leetcode-cn.com/problems/regular-expression-matching/
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

//打家劫舍 III
//https://leetcode-cn.com/problems/house-robber-iii/
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

//打家劫舍 II
//https://leetcode-cn.com/problems/house-robber-ii/
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

//打家劫舍
//https://leetcode-cn.com/problems/house-robber/
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

//买买卖股票的最佳时机 IV
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/
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

//买卖股票的最佳时机 III
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/
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

//买卖股票的最佳时机 II
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/
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

//买卖股票的最佳时机
//https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/
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
//https://leetcode-cn.com/problems/remove-covered-intervals/
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

//区间列表的交集
//https://leetcode-cn.com/problems/interval-list-intersections/
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

//合并区间
//https://leetcode-cn.com/problems/merge-intervals/
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

//无重叠区间
//https://leetcode-cn.com/problems/non-overlapping-intervals/
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
