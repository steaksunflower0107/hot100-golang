package main

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

func main() {
	// test area
}

func restoreIpAddresses(s string) []string {
	res := make([]string, 0)
	path := make([]string, 4)

	var dfs func(start, id int)

	dfs = func(start, id int) {
		if id > 0 {
			num, _ := strconv.Atoi(path[id-1])
			if num > 255 {
				return
			}

			if len(path[id-1]) > 1 && path[id-1][0] == '0' {
				return
			}
		}

		if id == 4 {
			var ans string
			var length int
			for i := range path {
				ans += path[i]
				length += len(path[i])
				if i != 3 {
					ans += "."
				}
			}

			if length == len(s) {
				res = append(res, ans)
			}
			return
		}

		for i := start; i < len(s) && i < start+3; i++ {
			path[id] = ""
			for m := start; m <= i; m++ {
				path[id] += string(s[m])
			}
			dfs(start+len(path[id]), id+1)
		}
	}

	dfs(0, 0)

	return res
}

func merge(intervals [][]int) [][]int {
	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i][0] < intervals[j][0] {
			return true
		}
		return false
	})

	res := make([][]int, 0)
	res = append(res, intervals[0])

	for i := 1; i < len(intervals); i++ {
		if intervals[i][0] <= res[len(res)-1][1] {
			if intervals[i][1] > res[len(res)-1][1] {
				res[len(res)-1][1] = intervals[i][1]
			}
		} else {
			res = append(res, intervals[i])
		}
	}

	return res
}

func merge2(nums1 []int, m int, nums2 []int, n int) {
	res := make([]int, m+n)
	var i, j int

	for i < m || j < n {
		var temp1, temp2 = math.MaxInt, math.MaxInt
		var temp int

		if i < m {
			temp1 = nums1[i]
		}

		if j < n {
			temp2 = nums2[j]
		}

		if temp1 > temp2 {
			temp = temp2
			j++
		} else {
			temp = temp1
			i++
		}
		res = append(res, temp)
	}

	copy(nums1, res)
}

func pathSum(root *TreeNode, targetSum int) int {
	var dfs func(node *TreeNode)
	var dfs1 func(i int, node *TreeNode)
	var ans int
	var path int

	layers := make(map[int][]*TreeNode)

	dfs1 = func(i int, node *TreeNode) {
		if node == nil {
			return
		}

		if _, ok := layers[i]; !ok {
			layers[i] = make([]*TreeNode, 0)
		} else {
			layers[i] = append(layers[i], node)
		}

		dfs1(i+1, node.Left)
		dfs1(i+1, node.Left)
	}

	dfs1(1, root)

	dfs = func(node *TreeNode) {
		if node != nil {
			return
		}

		path += root.Val

		if targetSum == path {
			ans++
		}

		dfs(root.Left)
		dfs(root.Right)

		path -= root.Val
	}

	for i := range layers {
		for _, node := range layers[i] {
			dfs(node)
		}
	}

	return ans
}

func valid(s string) bool {
	var left, right int
	for left < right {
		if s[left] != s[right] {
			return false
		}
		left++
		right--
	}

	return true
}

func exist(board [][]byte, word string) bool {
	boardBool := make([][]bool, len(board))
	for i := 0; i < len(board); i++ {
		boardBool[i] = make([]bool, len(board[0]))
	}
	var ans bool
	var temp string

	var dfs func(x, y, i int)
	dfs = func(x, y, i int) {
		if i == len(word) {
			return
		}

		if board[x][y] == word[i] {
			boardBool[x][y] = true
			temp += string(word[i])
			if temp == word {
				ans = true
				return
			}
			var added bool

			if i+1 < len(word) {
				if x+1 < len(board) && !boardBool[x+1][y] && board[x+1][y] == word[i+1] {
					dfs(x+1, y, i+1)
					temp = temp[:len(temp)-1]
					boardBool[x][y] = false
					added = true
				}

				if x-1 > 0 && !boardBool[x-1][y] && board[x-1][y] == word[i+1] {
					dfs(x-1, y, i+1)
					temp = temp[:len(temp)-1]
					boardBool[x][y] = false
					added = true
				}

				if y+1 < len(board[0]) && !boardBool[x][y+1] && board[x][y+1] == word[i+1] {
					dfs(x, y+1, i+1)
					temp = temp[:len(temp)-1]
					boardBool[x][y] = false
					added = true
				}

				if y-1 > 0 && !boardBool[x][y-1] && board[x][y-1] == word[i+1] {
					dfs(x, y-1, i+1)
					temp = temp[:len(temp)-1]
					boardBool[x][y] = false
					added = true
				}
			}

			if !added && len(temp) > 0 {
				temp = temp[:len(temp)-1]
			}
		}
	}

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			dfs(i, j, 0)
			if ans == true {
				return true
			}
		}
	}

	return false
}

func generateParenthesis(n int) []string {
	res := make([]string, 0)
	var path string
	left, right := n, n

	var dfs func(i int)
	dfs = func(i int) {
		if i == n*2 {
			res = append(res, path)
			return
		}

		if left > 0 && left <= right {
			path += "("
			left--
			dfs(i + 1)
			left++
			path = path[:len(path)-1]
		}

		if right > 0 && left <= right {
			path += ")"
			right--
			dfs(i + 1)
			right++
			path = path[:len(path)-1]
		}
	}

	dfs(0)

	return res
}

func combinationSum(candidates []int, target int) [][]int {
	res := make([][]int, 0)
	var dfs func(i, target int)
	path := make([]int, 0)

	dfs = func(i, target int) {
		fmt.Println(path)
		if target == 0 {
			res = append(res, append([]int(nil), path...))
			return
		}

		if target < 0 {
			return
		}

		for j := i; j < len(candidates); j++ {
			path = append(path, candidates[j])
			dfs(j, target-candidates[j])
			path = path[:len(path)-1]
		}
	}
	dfs(0, target)

	return res
}

var mapping = [...]string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}

func letterCombinations(digits string) []string {
	chars := make([]string, 0)
	res := make([]string, 0)

	for i := range digits {
		num, _ := strconv.Atoi(string(digits[i]))
		chars = append(chars, mapping[num])
	}

	var dfs func(i int)
	var path string

	dfs = func(i int) {
		if i == len(digits) {
			res = append(res, path)
			return
		}

		for j := 0; j < len(chars[i]); j++ {
			path += string(chars[i][j])
			dfs(i + 1)
			path = path[:len(path)-1]
		}
	}
	dfs(0)

	return res
}

func subsets(nums []int) [][]int {
	res := make([][]int, 0)

	var dfs func(i int)
	path := make([]int, 0)
	index := make([]int, 0)
	index = append(index, -1)

	dfs = func(i int) {
		res = append(res, append([]int(nil), path...))

		if i == len(nums) {
			return
		}

		for j := i; j < len(nums); j++ {
			if j > index[len(index)-1] {
				path = append(path, nums[j])
				index = append(index, j)
				dfs(i + 1)
				path = path[:len(path)-1]
				index = index[:len(index)-1]
			}
		}
	}
	dfs(0)

	return res
}

func permute(nums []int) [][]int {
	res := make([][]int, 0)
	var dfs func(i int)
	path := make([]int, 0)
	index := make(map[int]struct{})

	dfs = func(i int) {
		if i == len(nums) {
			res = append(res, append([]int(nil), path...))
			return
		}

		for j := 0; j < len(nums); j++ {
			if _, ok := index[j]; !ok {
				path = append(path, nums[j])
				index[j] = struct{}{}
				dfs(i + 1)
				delete(index, j)
				path = path[:len(path)-1]
			}
		}
	}

	dfs(0)

	return res
}

func longestValidParentheses(s string) int {
	stack := make([]int, 0)
	stack = append(stack, -1)
	var ans int

	for i := range s {
		if s[i] == '(' {
			stack = append(stack, i)
		} else {
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				stack = append(stack, i)
			} else {
				ans = max(ans, i-stack[len(stack)-1])
			}
		}
	}

	return ans
}

func canPartition(nums []int) bool {
	n := len(nums)

	if n%2 != 0 {
		return false
	}

	var sum, maxNum int
	for _, v := range nums {
		sum += v
		if v > maxNum {
			maxNum = v
		}
	}

	var target = sum / 2

	if maxNum > target {
		return false
	}
	// dp[i][j]表示从数组的[0,i]下标范围内选取若干个正整数，是否存在一种和为j的方案
	dp := make([][]bool, n)

	for i := range dp {
		dp[i] = make([]bool, target+1)
	}

	for i := 0; i < n; i++ {
		dp[i][0] = true
	}

	dp[0][nums[0]] = true
	for i := 1; i < n; i++ {
		v := nums[i]
		for j := 1; j <= target; j++ {
			if j >= v {
				dp[i][j] = dp[i-1][j] || dp[i-1][j-v]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}

	return dp[n-1][target]
}

func maxProduct(nums []int) int {
	maxDp, minDp, ans := nums[0], nums[0], nums[0]

	for i := 1; i < len(nums); i++ {
		mx, mn := maxDp, minDp
		maxDp = max(mx*nums[i], max(nums[i], mn*nums[i]))
		minDp = min(mn*nums[i], min(nums[i], mx*nums[i]))
		ans = max(maxDp, ans)
	}

	return ans
}

func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	var ans int

	for i := 1; i < len(nums); i++ {
		var temp int
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] {
				temp = max(temp, dp[j])
			}
		}
		dp[i] = temp + 1
		ans = max(ans, dp[i])
	}

	return dp[len(dp)-1]
}

func wordBreak(s string, wordDict []string) bool {
	dp := make([]bool, len(s)+1)
	dp[0] = true
	set := make(map[string]bool)

	for _, word := range wordDict {
		set[word] = true
	}

	for i := 1; i < len(s)+1; i++ {
		for j := 0; j < i; j++ {
			if dp[j] && set[s[j:i]] {
				dp[i] = true
				break
			}
		}
	}

	//fmt.Println(dp)

	return dp[len(dp)-1]
}

func coinChange(coins []int, amount int) int {
	dp := make([]int, amount+1)
	for i := range dp {
		dp[i] = math.MaxInt
	}

	dp[0] = 0

	for i := 1; i <= amount; i++ {
		for j := 0; j < len(coins); j++ {
			if i >= coins[j] && dp[i-coins[j]] != math.MaxInt {
				dp[i] = min(dp[i-coins[j]]+1, dp[i])
			}
		}
	}

	if dp[amount] == math.MaxInt {
		return -1
	}

	return dp[amount]
}

func numSquares(n int) int {
	dp := make([]int, n+1)
	for i := range dp {
		dp[i] = math.MaxInt
	}

	dp[0] = 0

	for i := 1; i <= n; i++ {
		for j := 1; j*j <= i; j++ {
			dp[i] = min(dp[i-j*j], dp[i])
		}
		dp[i] += 1
	}

	return dp[n]
}

func rob(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])

	for i := 2; i < len(nums); i++ {
		if nums[i]+dp[i-2] > dp[i-1] {
			dp[i] = dp[i-2] + nums[i]
		} else {
			dp[i] = dp[i-1]
		}
	}
	return dp[len(dp)-1]
}

func generate(numRows int) [][]int {
	ans := make([][]int, numRows)

	for i := 0; i < len(ans); i++ {
		if i == 0 || i == 1 {
			for j := 0; j < i+1; j++ {
				ans[i] = append(ans[i], 1)
			}
			continue
		} else {
			ans[i] = make([]int, i+1)
			ans[i][0] = 1
			ans[i][len(ans[i])-1] = 1
		}
	}

	for i := 1; i < len(ans)-1; i++ {
		var left, right = 0, 1
		for j := 1; j < len(ans[i+1])-1; j++ {
			if ans[i+1][j] == 0 {
				ans[i+1][j] = ans[i][left] + ans[i][right]
				left++
				right++
			}
		}
	}

	return ans
}

type MinStack struct {
	stack    []int
	minStack []int
}

func ConstructorMinStack() MinStack {
	stack := make([]int, 0)
	minStack := make([]int, 0)
	minStack = append(minStack, math.MaxInt)
	return MinStack{
		stack:    stack,
		minStack: minStack,
	}
}

func (this *MinStack) Push(val int) {
	this.stack = append(this.stack, val)
	this.minStack = append(this.minStack, min(this.minStack[len(this.minStack)-1], val))
}

func (this *MinStack) Pop() {
	this.minStack = this.minStack[:len(this.minStack)-1]
	this.stack = this.stack[:len(this.stack)-1]
}

func (this *MinStack) Top() int {
	return this.stack[len(this.stack)-1]
}

func (this *MinStack) GetMin() int {
	return this.minStack[len(this.minStack)-1]
}

func isValid(s string) bool {
	if len(s)%2 != 0 {
		return false
	}

	matchMap := map[byte]byte{
		'}': '{',
		')': '(',
		']': '[',
	}

	stack := make([]byte, 0)

	for i := 0; i < len(s); i++ {
		if _, ok := matchMap[s[i]]; ok {
			if len(stack) == 0 || stack[len(stack)-1] != matchMap[s[i]] {
				return false
			} else {
				stack = stack[:len(stack)-1]
			}
		} else {
			stack = append(stack, s[i])
		}
	}

	return len(stack) == 0
}

func searchRange(nums []int, target int) []int {
	left, right := 0, len(nums)-1
	var found bool

	for left < right {
		if nums[left] == target && nums[right] == target {
			found = true
			break
		}

		if nums[left] != target {
			left++
		}

		if nums[right] != target {
			right--
		}
	}

	if !found {
		return []int{-1, -1}
	}

	return []int{left, right}
}

func searchTwoMatrix(matrix [][]int, target int) bool {
	length, width := len(matrix), len(matrix[0])
	var row int

	for row < length-1 {
		if matrix[row][0] == target || matrix[row][width-1] == target {
			return true
		}

		if target > matrix[row][0] && target < matrix[row][width-1] {
			break
		} else if target > matrix[row][width-1] && target < matrix[row+1][0] {
			return false
		}

		row++
	}

	left, right := 0, width-1

	for left <= right {
		mid := (left + right) / 2

		if matrix[row][mid] == target {
			return true
		} else if matrix[row][mid] > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	return false
}

func searchInsert(nums []int, target int) int {
	ans := -1
	left, right := 0, len(nums)-1

	for left <= right {
		mid := (left + right) / 2
		if nums[mid] == target {
			ans = mid
			break
		} else if nums[mid] > target {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	var i, num int
	if ans == -1 {
		for i, num = range nums {
			if num > target {
				return i
			}
		}

		return i + 1
	}

	return ans
}

func leastInterval(tasks []byte, n int) int {
	return 0
}

func longestCommonSubsequence(text1 string, text2 string) int {
	m, n := len(text1), len(text2)
	dp := make([][]int, m+1)

	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	for i, c1 := range text1 {
		for j, c2 := range text2 {
			if c1 == c2 {
				dp[i+1][j+1] = dp[i][j] + 1
			} else {
				dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
			}
		}
	}

	return dp[m][n]
}

func canJump(nums []int) bool {
	var cover int
	for i := 0; i <= cover; i++ {
		cover = max(cover, i+nums[i])
		if cover > len(nums) {
			return true
		}
	}

	return false
}

func levelOrderBottom(root *TreeNode) [][]int {
	var dfs func(node *TreeNode, i int)
	layers := make(map[int][]int)
	var depth int
	dfs = func(node *TreeNode, i int) {
		if node == nil {
			return
		}

		if _, ok := layers[i]; !ok {
			layers[i] = make([]int, 0)
		}

		layers[i] = append(layers[i], node.Val)

		depth = max(i, depth)

		dfs(node.Left, i+1)
		dfs(node.Right, i+1)
	}

	ans := make([][]int, depth)
	for depth > 0 {
		ans[depth] = layers[depth]
		depth--
	}

	return ans
}

func dailyTemperatures(temperatures []int) []int {
	ans := make([]int, len(temperatures))
	next := false
	for i := len(temperatures) - 1; i >= 0; i-- {
		j := i + 1
		for j < len(temperatures) && temperatures[j] <= temperatures[i] {
			if ans[j] == 0 && temperatures[i] > temperatures[j] {
				ans[i] = 0
				next = true
				break
			}
			j++
		}

		if next {
			next = false
			continue
		}

		if j == len(temperatures) {
			ans[i] = 0
		} else {
			ans[i] = j - i
		}
	}

	ans[len(temperatures)-1] = 0

	return ans
}

func sortedArrayToBST(nums []int) *TreeNode {
	return build(nums, 0, len(nums)-1)
}

func build(nums []int, left, right int) *TreeNode {
	if left > right {
		return nil
	}

	mid := (left + right + 1) / 2
	root := &TreeNode{Val: nums[mid]}

	root.Left = build(nums, left, mid-1)
	root.Right = build(nums, mid+1, right)

	return root
}

func levelOrder(root *TreeNode) [][]int {
	layers := make(map[int][]int)
	var layerCount int
	res := make([][]int, 0)

	var dfs func(i int, root *TreeNode)
	dfs = func(i int, root *TreeNode) {
		if root == nil {
			return
		}

		if _, ok := layers[i]; !ok {
			layers[i] = make([]int, 0)
			layers[i] = append(layers[i], root.Val)
		} else {
			layers[i] = append(layers[i], root.Val)
		}

		dfs(i+1, root.Left)
		dfs(i+1, root.Right)
	}
	dfs(1, root)

	layerCount = len(layers)
	for i := 1; i <= layerCount; i++ {
		res = append(res, layers[i])
	}

	return res
}

func diameterOfBinaryTree(root *TreeNode) int {
	res := 0
	var dfs func(root *TreeNode) int
	dfs = func(root *TreeNode) int {
		if root == nil {
			return 0
		}

		left := dfs(root.Left)
		right := dfs(root.Right)

		res = max(res, left+right)

		return max(left, right) + 1
	}

	dfs(root)

	return res
}

type LRUCache struct {
	capacity   int
	cache      map[int]*LRUNode
	head, tail *LRUNode
}

type LRUNode struct {
	key, value int
	pre, next  *LRUNode
}

func InitLRUNode(key, value int) *LRUNode {
	return &LRUNode{
		key:   key,
		value: value,
	}
}

func Constructor(capacity int) LRUCache {
	cache := make(map[int]*LRUNode)

	res := LRUCache{
		cache:    cache,
		capacity: capacity,
		head:     InitLRUNode(0, 0),
		tail:     InitLRUNode(0, 0),
	}

	res.head.next = res.tail
	res.tail.pre = res.head

	return res
}

func (l *LRUCache) moveToHead(node *LRUNode) {
	l.removeNode(node)
	l.addToHead(node)
}

func (l *LRUCache) removeNode(node *LRUNode) {
	node.pre.next = node.next
	node.next.pre = node.pre
}

func (l *LRUCache) addToHead(node *LRUNode) {
	node.next = l.head.next.next
	l.head.next.next.pre = node
	l.head.next = node
	node.pre = l.head
}

func (l *LRUCache) removeTail() int {
	node := l.tail.pre
	l.tail.pre = node.pre
	node.pre.next = l.tail

	return node.key
}

func (this *LRUCache) Get(key int) int {
	if node, ok := this.cache[key]; ok {
		this.moveToHead(node)
		return node.value
	} else {
		return -1
	}
}

func (this *LRUCache) Put(key int, value int) {
	if node, ok := this.cache[key]; ok {
		node.value = value
		this.moveToHead(node)
	} else {
		node = InitLRUNode(key, value)
		this.cache[key] = node
		this.addToHead(node)
		if len(this.cache) > this.capacity {
			key = this.removeTail()
			delete(this.cache, key)
		}
	}
}

func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	Left := invertTree(root.Left)
	Right := invertTree(root.Right)

	root.Left = Right
	root.Right = Left

	return root
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}

	ans, temp := 1, 1
	var dfs func(node *TreeNode)

	dfs = func(node *TreeNode) {
		if node.Right == nil && node.Left == nil {
			ans = max(ans, temp)
			return
		}

		if node.Left != nil {
			temp++
			dfs(node.Left)
			temp--
		}

		if node.Right != nil {
			temp++
			dfs(node.Right)
			temp--
		}
	}

	dfs(root)

	return ans
}

func inorderTraversal(root *TreeNode) []int {
	ans := make([]int, 0)
	var inorder func(node *TreeNode)

	inorder = func(node *TreeNode) {
		if node == nil {
			return
		}

		inorder(node.Left)

		ans = append(ans, node.Val)

		inorder(node.Right)
	}

	inorder(root)

	return ans
}

func mergeKLists(lists []*ListNode) *ListNode {
	nodes := make([]int, 0)
	for _, head := range lists {
		for head != nil {
			nodes = append(nodes, head.Val)
			head = head.Next
		}
	}

	if len(nodes) == 0 {
		return nil
	}

	sort.Ints(nodes)

	ans := &ListNode{Val: nodes[0]}
	curr := ans
	for i := 1; i < len(nodes); i++ {
		newNode := &ListNode{Val: nodes[i]}
		curr.Next = newNode
		curr = curr.Next
	}

	return ans
}

func sortList(head *ListNode) *ListNode {
	if head == nil {
		return head
	}

	nodes := make([]int, 0)
	for head != nil {
		nodes = append(nodes, head.Val)
		head = head.Next
	}

	sort.Ints(nodes)

	ans := &ListNode{Val: nodes[0]}
	curr := ans

	for i := 1; i < len(nodes); i++ {
		newNode := &ListNode{Val: nodes[i]}
		curr.Next = newNode
		curr = curr.Next
	}

	return ans
}

var cacheNode map[*Node]*Node

func deepCopy(head *Node) *Node {
	if head == nil {
		return nil
	}

	if node, ok := cacheNode[head]; ok {
		return node
	}

	newNode := &Node{Val: head.Val}
	cacheNode[head] = newNode
	newNode.Next = deepCopy(head.Next)
	newNode.Random = deepCopy(head.Random)

	return newNode
}

func copyRandomList(head *Node) *Node {
	cacheNode = make(map[*Node]*Node)
	return deepCopy(head)
}

func reverseKGroup(head *ListNode, k int) *ListNode {
	if getLength(head) < k {
		return head
	}

	var pre *ListNode
	newHead := head

	count := k
	for count != 0 {
		newHead = newHead.Next
		count--
	}

	count = k
	for count != 0 {
		next := head.Next
		if count == k {
			head.Next = reverseKGroup(newHead, k)
		} else {
			head.Next = pre
		}

		pre = head
		head = next
		count--
	}

	return pre
}

func getLength(head *ListNode) (ans int) {
	for head != nil {
		ans++
		head = head.Next
	}
	return
}

func swapPairsByRecursion(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}

	newHead := head.Next

	head.Next = swapPairsByRecursion(head.Next.Next)
	newHead.Next = head

	return newHead
}

func swapPairs(head *ListNode) *ListNode {

	if head == nil {
		return nil
	}

	if head.Next == nil {
		return head
	}

	nodes := make([]int, 0)

	for head != nil {
		nodes = append(nodes, head.Val)
		head = head.Next
	}

	for i := 0; i < len(nodes)-1; i += 2 {
		nodes[i], nodes[i+1] = nodes[i+1], nodes[i]
	}

	head = &ListNode{Val: nodes[0]}
	curr := head

	for i := 1; i < len(nodes); i++ {
		newNode := &ListNode{Val: nodes[i]}
		curr.Next = newNode
		curr = newNode
	}

	return head
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	nodes := make([]*ListNode, 0)

	for head != nil {
		nodes = append(nodes, head)
		head = head.Next
	}

	length := len(nodes)

	if length == 1 {
		return nil
	}

	if n == 1 {
		nodes[length-n-1].Next = nil
	} else if n == length {
		nodes[0] = nodes[1]
	} else {
		nodes[length-n-1].Next = nodes[length-n+1]
	}

	return nodes[0]
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var entry int
	nodes := make([]*ListNode, 0)

	for l1 != nil || l2 != nil {
		var temp1, temp2 int
		if l1 != nil {
			temp1 = l1.Val
			l1 = l1.Next
		}

		if l2 != nil {
			temp2 = l2.Val
			l2 = l2.Next
		}

		newVal := temp1 + temp2
		if entry != 0 {
			newVal += entry
			entry = 0
		}

		if newVal >= 10 {
			entry = 1
			newVal -= 10
		}

		newNode := &ListNode{Val: newVal}
		nodes = append(nodes, newNode)
	}

	if entry != 0 {
		nodes = append(nodes, &ListNode{Val: entry})
	}

	for i := 0; i < len(nodes)-1; i++ {
		nodes[i].Next = nodes[i+1]
	}

	return nodes[0]
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	if list1 == nil {
		return list2
	}

	if list2 == nil {
		return list1
	}

	if list1.Val < list2.Val {
		list1.Next = mergeTwoLists(list1.Next, list2)
		return list1
	} else {
		list2.Next = mergeTwoLists(list1, list2.Next)
		return list2
	}
}

func detectCycle(head *ListNode) *ListNode {
	seen := make(map[*ListNode]struct{})
	for head != nil {
		if _, ok := seen[head]; ok {
			return head
		} else {
			seen[head] = struct{}{}
		}
		head = head.Next
	}
	return nil
}

func hasCycle(head *ListNode) bool {
	seen := make(map[*ListNode]struct{})
	for head != nil {
		if _, ok := seen[head]; ok {
			return true
		} else {
			seen[head] = struct{}{}
		}
		head = head.Next
	}

	return false
}

func isPalindrome(head *ListNode) bool {
	nums := make([]int, 0)
	for head != nil {
		nums = append(nums, head.Val)
		head = head.Next
	}

	left, right := 0, len(nums)-1
	for left != right && left < right {
		if nums[left] != nums[right] {
			return false
		}
		left++
		right--
	}

	return true
}

func reverseList(head *ListNode) *ListNode {

	var pre *ListNode

	for head != nil {
		next := head.Next
		head.Next = pre

		pre = head
		head = next
	}

	return pre
}

func getIntersectionNode(headA, headB *ListNode) *ListNode {
	setA := make(map[*ListNode]struct{})

	for temp := headA; temp != nil; temp = temp.Next {
		setA[temp] = struct{}{}
	}

	for temp := headB; temp != nil; temp = temp.Next {
		if _, ok := setA[temp]; ok {
			return temp
		}
	}

	return nil
}

func searchMatrix(matrix [][]int, target int) bool {
	x, y := 0, len(matrix[0])-1

	for x < len(matrix) && y >= 0 {
		if matrix[x][y] == target {
			return true
		}

		if matrix[x][y] > target {
			y--
		} else {
			x++
		}
	}

	return false
}

func rotatePicture(matrix [][]int) {
	ans := make([][]int, len(matrix))
	for i := range ans {
		ans[i] = make([]int, len(matrix[0]))
	}

	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			ans[j][len(matrix)-1-i] = matrix[i][j]
		}
	}

	copy(matrix, ans)

	// 0,1 -> 1,2
	// 0,2 -> 2,2
	// 0,3 -> 3,2

	// 2,1 -> 1,2
	// 2,2 -> 2,2
	// 2,3 -> 3,2

	// 1,1 -> 4,1
	// 1,2 -> 4,2
}

func spiralOrder(matrix [][]int) []int {
	ans := make([]int, 0)

	checked := make([][]bool, len(matrix))
	for i := range checked {
		checked[i] = make([]bool, len(matrix[0]))
	}

	var x, y int

	for !finish(checked) {
		for y < len(matrix[0]) && checked[x][y] == false {
			ans = append(ans, matrix[x][y])
			checked[x][y] = true
			y++
		}

		y--
		x++

		for x < len(matrix) && checked[x][y] == false {
			ans = append(ans, matrix[x][y])
			checked[x][y] = true
			x++
		}
		x--
		y--

		for y >= 0 && checked[x][y] == false {
			ans = append(ans, matrix[x][y])
			checked[x][y] = true
			y--
		}
		y++
		x--

		for x >= 0 && checked[x][y] == false {
			ans = append(ans, matrix[x][y])
			checked[x][y] = true
			x--
		}

		x++
		y++
	}

	return ans
}

func finish(checked [][]bool) bool {
	for i := 0; i < len(checked); i++ {
		for j := 0; j < len(checked[0]); j++ {
			if checked[i][j] == false {
				return false
			}
		}
	}

	return true
}

func setZeroes(matrix [][]int) {
	zeroes := make([][]int, 0)

	// 行
	for i := 0; i < len(matrix); i++ {
		// 列
		for j := 0; j < len(matrix[0]); j++ {
			if matrix[i][j] == 0 {
				zeroes = append(zeroes, []int{i, j})
			}
		}
	}

	for _, zero := range zeroes {
		setZero(matrix, zero[0], zero[1])
	}
}

func setZero(matrix [][]int, x, y int) {
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			if i == x || j == y {
				matrix[i][j] = 0
			}
		}
	}
}

func firstMissingPositive(nums []int) int {
	set := make(map[int]struct{})
	var i = 1

	for _, v := range nums {
		set[v] = struct{}{}
	}

	for i <= len(nums)+1 {
		if _, ok := set[i]; !ok {
			return i
		}
		i++
	}

	return i
}

func productExceptSelf(nums []int) []int {
	ans := make([]int, len(nums))

	left := make([]int, len(nums))
	left[0] = 1
	right := make([]int, len(nums))
	right[len(right)-1] = 1

	for i := 1; i < len(nums); i++ {
		left[i] = left[i-1] * nums[i-1]
	}

	for i := len(nums) - 2; i >= 0; i-- {
		right[i] = right[i+1] * nums[i+1]
	}

	for i := range nums {
		ans[i] = left[i] * right[i]
	}

	return ans
}

func rotate(nums []int, k int) {

	if len(nums) <= 1 {
		return
	}

	for len(nums) < k {
		k = k - len(nums)
	}

	ans := make([]int, 0)
	for i := len(nums) - k; i < len(nums); i++ {
		ans = append(ans, nums[i])
	}

	for i := 0; i < len(nums)-k; i++ {
		ans = append(ans, nums[i])
	}

	copy(nums, ans)
}

func merge1(intervals [][]int) [][]int {

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})

	res := [][]int{intervals[0]}

	for i := 1; i < len(intervals); i++ {
		if res[len(res)-1][1] >= intervals[i][0] && res[len(res)-1][1] < intervals[i][1] {
			res = append(res, []int{res[len(res)-1][0], intervals[i][1]})
			res = append(res[:len(res)-2], res[len(res)-1:]...)
		} else if res[len(res)-1][1] >= intervals[i][1] {
			continue
		} else {
			res = append(res, intervals[i])
		}
	}

	return res
}

func maxSubArray(nums []int) int {
	ans := nums[0]

	for i := 1; i < len(nums); i++ {
		if nums[i]+nums[i-1] > nums[i] {
			nums[i] = nums[i-1] + nums[i]
		}

		ans = max(nums[i], ans)
	}

	return ans
}

func minWindow(s string, t string) string {
	if len(t) > len(s) {
		return ""
	}

	var ans string

	tMap := make(map[uint8]int)
	sMap := make(map[uint8]int)

	for i, _ := range t {
		tMap[t[i]]++
	}

	lp, rp := 0, 0
	firstTime := true
	sub := ""

	for rp < len(s) {

		for rp < len(s) && rp-lp+1 <= len(t) {
			sMap[s[rp]]++
			sub += string(s[rp])
			rp++
		}

		for rp < len(s) && !match(tMap, sMap) {
			sMap[s[rp]]++
			sub += string(s[rp])
			rp++
		}

		for match(tMap, sMap) {
			if firstTime {
				ans = sub
				firstTime = false
			}

			if len(sub) < len(ans) {
				ans = sub
			}

			lp++
			if lp > 0 {
				sMap[s[lp-1]]--
				sub = sub[1:]
			}
		}
	}

	return ans
}

func match(tMap map[uint8]int, sMap map[uint8]int) bool {
	for k, v := range tMap {
		if sMap[k] < v {
			return false
		}
	}

	return true
}

func maxSlidingWindow(nums []int, k int) []int {
	if len(nums) == 0 {
		return nil
	}

	res := make([]int, 0)
	deque := make([]int, 0)

	for i := 0; i < len(nums); i++ {
		if len(deque) > 0 && deque[0] < i-k+1 {
			deque = deque[1:]
		}

		for len(deque) > 0 && nums[deque[len(deque)-1]] < nums[i] {
			deque = deque[:len(deque)-1]
		}

		deque = append(deque, i)

		if i-k+1 >= 0 {
			res = append(res, nums[deque[0]])
		}
	}

	return res
}

func subarraySum(nums []int, k int) int {
	var ans int
	var rk int
	for i := 0; i < len(nums); i++ {
		if nums[i] == k {
			ans++
			continue
		}

		rk = i + 1
		temp := nums[i]
		for rk < len(nums) {
			temp += nums[rk]
			if temp == k {
				ans++
			}
			rk++
		}
	}

	return ans
}

func findAnagrams(s, p string) (ans []int) {
	sLen, pLen := len(s), len(p)

	if sLen < pLen {
		return
	}

	var pCount, sCount [26]int

	for i, ch := range p {
		pCount[ch-'a']++
		sCount[s[i]-'a']++
	}

	if pCount == sCount {
		ans = append(ans, 0)
	}

	for i, ch := range s[:sLen-pLen] {
		sCount[ch-'a']--
		sCount[s[i+pLen]-'a']++

		if sCount == pCount {
			ans = append(ans, i+1)
		}
	}

	return
}

func lengthOfLongestSubstring(s string) int {
	var ans int
	set := make(map[uint8]int)
	rk := -1

	for i, _ := range s {
		if i != 0 {
			delete(set, s[i-1])
		}

		for rk+1 < len(s) && set[s[rk+1]] == 0 {
			set[s[rk+1]]++
			rk++
		}

		ans = max(ans, rk-i+1)
	}

	return ans
}

func groupAnagrams(strs []string) (res [][]string) {
	myMap := make(map[string][]string)
	for _, word := range strs {
		charSlice := strings.Split(word, "")
		sort.Strings(charSlice)
		newWord := strings.Join(charSlice, "")

		myMap[newWord] = append(myMap[newWord], word)

	}

	for _, v := range myMap {
		res = append(res, v)
	}
	return
}

func longestConsecutive(nums []int) (ans int) {
	numsSet := make(map[int]int)
	for _, num := range nums {
		numsSet[num] = 0
	}

	exist := func(num int) bool {
		if _, ok := numsSet[num+1]; ok {
			return true
		}
		return false
	}

	for _, num := range nums {
		if _, ok := numsSet[num-1]; ok {
			continue
		}
		temp := 1
		for exist(num) == true {
			num++
			exist(num)
			temp++
		}
		ans = max(ans, temp)
	}

	return
}

func moveZeroes(nums []int) {
	left, right := 0, 0

	for right < len(nums) {
		if nums[right] != 0 {
			nums[left], nums[right] = nums[right], nums[left]
			left++
		}
		right++
	}

	return
}

func maxArea(height []int) int {
	var ans int
	left, right := 0, len(height)-1
	for left < right {
		temp := (right - left) * min(height[left], height[right])
		ans = max(ans, temp)
		if height[right] > height[left] {
			left++
		} else {
			right--
		}
	}
	return ans
}

func threeSum(nums []int) (res [][]int) {
	if len(nums) < 3 {
		return
	}

	sort.Ints(nums)

	for i := 0; i < len(nums)-1; i++ {
		if nums[i] > 0 {
			return
		}

		if i > 0 && nums[i] == nums[i-1] {
			continue
		}

		left, right := i+1, len(nums)-1

		for left < right {
			if nums[left]+nums[right]+nums[i] == 0 {
				res = append(res, []int{nums[i], nums[left], nums[right]})
				for left < right && nums[left] == nums[left+1] {
					left++
				}
				for left < right && nums[right] == nums[right-1] {
					right--
				}
				left++
				right--
			} else if nums[left]+nums[right]+nums[i] < 0 {
				left++
			} else {
				right--
			}
		}
	}

	return
}

func trap(height []int) int {
	var ans int
	leftMax := make([]int, len(height))
	leftMax[0] = 0
	rightMax := make([]int, len(height))
	rightMax[len(rightMax)-1] = 0

	for i := 1; i < len(height)-1; i++ {
		if height[i] > leftMax[i-1] {
			leftMax[i] = height[i]
		} else {
			leftMax[i] = leftMax[i-1]
		}
	}

	for i := len(height) - 2; i >= 0; i-- {
		if height[i] > rightMax[i+1] {
			rightMax[i] = height[i]
		} else {
			rightMax[i] = rightMax[i+1]
		}
	}

	fmt.Println(leftMax)
	fmt.Println(rightMax)

	for i, h := range height {
		ans += min(rightMax[i], leftMax[i]) - h
	}

	return ans
}

func createLinkedList(arr []int) *ListNode {
	if len(arr) == 0 {
		return nil
	}

	head := &ListNode{Val: arr[0]}
	current := head

	for i := 1; i < len(arr); i++ {
		newNode := &ListNode{Val: arr[i]}
		current.Next = newNode
		current = newNode
	}

	return head
}

func printList(node *ListNode) {
	for node != nil {
		fmt.Println(node.Val)
		node = node.Next
	}

	return
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

type ListNode struct {
	Val  int
	Next *ListNode
}

type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

func max(i, j int) int {
	if i > j {
		return i
	}
	return j
}

func min(x, y int) int {
	if x > y {
		return y
	}
	return x
}

func buildTree(input string) *TreeNode {
	tokens := strings.Fields(input)
	return buildTreeHelper(&tokens)
}

func buildTreeHelper(tokens *[]string) *TreeNode {
	if len(*tokens) == 0 {
		return nil
	}

	valStr := (*tokens)[0]
	*tokens = (*tokens)[1:]

	if valStr == "#" {
		return nil
	}

	val, err := strconv.Atoi(valStr)
	if err != nil {
		panic(err)
	}

	node := &TreeNode{Val: val}
	node.Left = buildTreeHelper(tokens)
	node.Right = buildTreeHelper(tokens)

	return node
}

func treeBuilder(nums []int) *TreeNode {
	treeNode := make([]*TreeNode, len(nums))

	for i := 0; i < len(nums); i++ {
		var node *TreeNode
		if nums[i] != -1 {
			node = &TreeNode{Val: nums[i]}
			treeNode[i] = node
		}
	}

	for i := 0; i*2+2 < len(nums); i++ {
		if treeNode[i] != nil {
			treeNode[i].Left = treeNode[i*2+1]
			treeNode[i].Right = treeNode[i*2+2]
		}
	}

	return treeNode[0]
}
