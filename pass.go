package main

// https://leetcode-cn.com/problems/list-of-depth-lcci/submissions/
func listOfDepth(tree *TreeNode) []*ListNode {
	treeNodes := make([]*TreeNode,0)
	listNodes := make([]*ListNode,0)

	if tree == nil {
		return listNodes
	}

	treeNodes = append(treeNodes,tree)

	for len(treeNodes) > 0 {
		nextTreeNodes := make([]*TreeNode,0)
		tmpNode := &ListNode{}
		headNode := tmpNode

		for i:=0;i<len(treeNodes);i++{
			tmpNode.Next = &ListNode{Val: treeNodes[i].Val}
			tmpNode = tmpNode.Next

			if treeNodes[i].Left != nil {
				nextTreeNodes = append(nextTreeNodes,treeNodes[i].Left)
			}
			if treeNodes[i].Right != nil {
				nextTreeNodes = append(nextTreeNodes,treeNodes[i].Right)
			}
		}

		listNodes = append(listNodes, headNode.Next)
		treeNodes = nextTreeNodes
	}

	return listNodes
}