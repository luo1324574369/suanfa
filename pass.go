package main

import (
	"math"
	"sort"
	"strings"
)

//子集
//https://leetcode-cn.com/problems/subsets/
func subsets(nums []int) [][]int {
	res := make([][]int,0)
	nl := len(nums)
	if nl == 0 {
		return res
	}
	var backtrack func(path []int, start int)
	backtrack = func(path []int, start int) {
		tmp := make([]int,len(path))
		copy(tmp,path)
		res = append(res,tmp)
		for i:=start+1;i<nl;i++ {
			tmp = append(tmp,nums[i])
			backtrack(tmp,i)
			tmp = tmp[:len(tmp)-1]
		}
	}
	backtrack([]int{},-1)
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

	isValid := func(path []int,row int,col int) bool {
		for i:=0;i<=row-1;i++{
			if path[i] == 1 << col {
				return false
			}
		}

		for i, j := row-1, col-1; i >= 0 && j >= 0; i,j = i-1,j-1 {
			if path[i] == 1 << j {
				return false
			}
		}

		for i, j := row-1, col+1; i >= 0 && j >=0 && j < nl; i,j = i-1,j+1 {
			if path[i] == 1 << j {
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

	path := make([]int,nl)

	var backtrack func(row int)
	backtrack = func(row int) {
		if row == nl {
			tmp := make([]int, nl)
			copy(tmp, path)
			res = append(res,tmp)
			return
		}

		for i:=0;i<nl;i++{
			if isValid(path,row,i) {
				path[row] = makeQ(nl,i)
				backtrack(row+1)
				path[row] = makeQ(nl,-1)
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

	isValid := func(path []string,row int,col int) bool {
		for i:=0;i<=row-1;i++{
			if path[i][col] == 'Q'{
				return false
			}
		}

		for i, j := row-1, col-1; i >= 0 && j >= 0; i,j = i-1,j-1 {
			if path[i][j] == 'Q' {
				return false
			}
		}

		for i, j := row-1, col+1; i >= 0 && j >=0 && j < nl; i,j = i-1,j+1 {
			if path[i][j] == 'Q' {
				return false
			}
		}
		return true
	}
	makeQ := func(totalLen int, pos int) string {
		if pos < 0 {
			return strings.Repeat(".",totalLen)
		}
		return strings.Repeat(".",pos) + "Q" + strings.Repeat(".",totalLen-pos-1)
	}

	path := make([]string,nl)
	for i:=0;i<nl;i++{
		path[i] = makeQ(nl,-1)
	}

	var backtrack func(row int)
	backtrack = func(row int) {
		if row == nl {
			tmp := make([]string, nl)
			copy(tmp, path)
			res = append(res,tmp)
			return
		}

		for i:=0;i<nl;i++{
			if isValid(path,row,i) {
				path[row] = makeQ(nl,i)
				backtrack(row+1)
				path[row] = makeQ(nl,0)
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

	useds := make([]bool,nl)
	sort.Ints(nums)
	var backtrack func(path []int,used []bool)
	backtrack = func(path []int, used []bool) {
		if len(path) == nl {
			tmp := make([]int, len(nums))
			copy(tmp, path)
			res = append(res,tmp)
			return
		}

		for i:=0;i<nl;i++{
			if i > 0 && nums[i] == nums[i-1] && !useds[i-1] {
				continue
			}

			if !used[i] {
				path = append(path,nums[i])
				used[i] = true

				backtrack(path,used)

				path = path[:len(path)-1]
				used[i] = false
			}
		}
	}

	backtrack(make([]int,0),useds)

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

	useds := make([]bool,nl)

	var backtrack func(path []int,used []bool)
	backtrack = func(path []int, used []bool) {
		if len(path) == nl {
			tmp := make([]int, len(nums))
			copy(tmp, path)
			res = append(res,tmp)
			return
		}

		for i:=0;i<nl;i++{
			if !used[i] {
				path = append(path,nums[i])
				used[i] = true

				backtrack(path,used)

				path = path[:len(path)-1]
				used[i] = false
			}
		}
	}

	backtrack(make([]int,0),useds)

	return res
}

//反转链表 II
//https://leetcode-cn.com/problems/reverse-linked-list-ii/
func reverseBetween(head *ListNode, m int, n int) *ListNode {
	if head == nil {
		return head
	}

	var rlN func(h *ListNode,n int) *ListNode
	var lastNext *ListNode
	rlN = func(h *ListNode, n int) *ListNode {
		if n == 1 {
			lastNext = h.Next
			return h
		}

		last := rlN(h.Next,n - 1)
		h.Next.Next = h
		h.Next = lastNext
		return last
	}

	if m == 1 {
		return rlN(head,n)
	}

	head.Next = reverseBetween(head.Next,m-1,n-1)

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
	res := make([]int,0)

	for i:=0;i<len(nums);i++{
		if i < k - 1 {
			dq.Push(nums[i])
		}else{
			dq.Push(nums[i])
			res = append(res,dq.Front())
			dq.Pop(nums[i-k+1])
		}
	}

	return res
}

//链表中的下一个更大节点
//https://leetcode-cn.com/problems/next-greater-node-in-linked-list/
func nextLargerNodes(head *ListNode) []int {
	ans := make([]int,0)
	stack := make([]int,0)

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
			ans = append(ans,0)
		}else{
			ans = append(ans,stack[len(stack)-1])
		}

		stack = append(stack,h.Val)
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
			tNew := &TreeNode{Val: val,Left: temp}
			root = tNew
			return root
		} else if temp.Right == nil{
			temp.Right = &TreeNode{Val: val}
			return root
		} else if temp.Right.Val >= val {
			temp = temp.Right
		} else if temp.Right.Val < val {
			temp.Right = &TreeNode{Val: val,Left: temp.Right}
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

	for i:=1;i<nl;i++{
		root = insertIntoMaxTree(root,nums[i])
	}

	return root
}

//正则表达式匹配
//https://leetcode-cn.com/problems/regular-expression-matching/
func isMatch2(s string, p string) bool {
	pl,sl := len(p),len(s)
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
		return isMatch(s,p[2:]) || (firstMatch && isMatch(s[1:],p))
	}

	return firstMatch && isMatch(s[1:],p[1:])
}

func isMatch(s string, p string) bool {
	dp := make([][]int,len(s)+1)
	for k,_ := range dp {
		dp[k] = make([]int,len(p)+1)
	}

	var isM func(si int,pj int) bool

	isM = func(si int,pj int) bool {
		if dp[si][pj] != 0 {
			return dp[si][pj] == 2
		}

		sl,pl := len(s[si:]),len(p[pj:])

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
			r := isM(si,pj+2) || (firstMatch && isM(si+1,pj))
			if r == true {
				dp[si][pj] = 2
			}else{
				dp[si][pj] = 1
			}
			return r
		}

		r := firstMatch && isM(si+1,pj+1)
		if r == true {
			dp[si][pj] = 2
		}else{
			dp[si][pj] = 1
		}
		return r
	}

	return isM(0,0)
}

//打家劫舍 III
//https://leetcode-cn.com/problems/house-robber-iii/
func rob3(root *TreeNode) int {
	var res []int
	var r func(node *TreeNode) []int
	r = func(node *TreeNode) []int {
		if node == nil {
			return []int{0,0}
		}
		l,r := r(node.Left),r(node.Right)
		selected := node.Val + l[1] + r[1]
		notSelected := max(l[0], l[1]) + max(r[0], r[1])
		return []int{selected, notSelected}
	}

	res = r(root)

	return max(res[0],res[1])
}

//打家劫舍 II
//https://leetcode-cn.com/problems/house-robber-ii/
func rob2(nums []int) int {
	nl := len(nums)

	if nl == 1 {
		return nums[0]
	}

	var r func(start int,end int) int

	r = func(start int, end int) int {
		pre1 := 0
		pre2 := 0
		for i:=end;i>=start;i--{
			temp := pre1
			pre1 = max(pre1, pre2+ nums[i])
			pre2 = temp
		}

		return pre1
	}

	return max(r(0,nl-2),r(1,nl-1))
}

//打家劫舍
//https://leetcode-cn.com/problems/house-robber/
func rob1(nums []int) int {
	nl := len(nums)

	pre1 := 0
	pre2 := 0
	for i:=nl-1;i>=0;i--{
		temp := pre1
		pre1 = max(pre1, pre2+ nums[i])
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
