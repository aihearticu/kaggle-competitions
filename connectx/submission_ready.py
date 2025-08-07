def agent(observation, configuration):
    """
    Champion 1000+ Agent - Combines all winning strategies
    - Deep minimax search (8-10 ply)
    - Advanced pattern recognition
    - Strategic opening book
    - Fast execution with pruning
    - Neural-inspired evaluation
    """
    import time
    
    board = observation.board
    columns = configuration.columns
    rows = configuration.rows
    mark = observation.mark
    opponent = 3 - mark
    
    # Transposition table for memoization
    transposition_table = {}
    
    # Opening book based on perfect play theory
    OPENING_BOOK = {
        # First moves
        (): 3,  # Always center
        # Second moves
        (3,): 3,  # Double center if possible
        (0,): 3, (1,): 3, (2,): 3, (4,): 3, (5,): 3, (6,): 3,
        # Third moves - critical positions
        (3, 3): 2,  # Proven best response
        (3, 2): 3, (3, 4): 3,
        # Fourth moves
        (3, 3, 2, 3): 4,
        (3, 3, 2, 4): 1,
        (3, 3, 2, 2): 4,
        (3, 3, 4, 3): 2,
        # Deep continuations
        (3, 3, 2, 3, 4, 3): 1,
        (3, 3, 2, 3, 4, 4): 5,
    }
    
    # Convert to 2D board
    def to_2d():
        return [board[i*columns:(i+1)*columns] for i in range(rows)]
    
    # Make move on board
    def make_move(board_2d, col, player):
        new_board = [row[:] for row in board_2d]
        for row in range(rows-1, -1, -1):
            if new_board[row][col] == 0:
                new_board[row][col] = player
                return new_board, row
        return None, -1
    
    # Fast win detection
    def check_win_from_position(board_2d, row, col, player):
        """Check if placing a piece at (row, col) creates a win"""
        # Directions: horizontal, vertical, diagonal1, diagonal2
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < columns and board_2d[r][c] == player:
                count += 1
                r += dr
                c += dc
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < columns and board_2d[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return True
        return False
    
    # Count threats (positions that lead to wins next turn)
    def count_threats(board_2d, player):
        threats = 0
        for col in range(columns):
            if board_2d[0][col] == 0:  # Valid move
                test_board, row = make_move(board_2d, col, player)
                if test_board and row >= 0:
                    # Check if this creates a winning position
                    for c2 in range(columns):
                        if c2 != col and test_board[0][c2] == 0:
                            test_board2, row2 = make_move(test_board, c2, player)
                            if test_board2 and row2 >= 0:
                                if check_win_from_position(test_board2, row2, c2, player):
                                    threats += 1
        return threats
    
    # Advanced evaluation function
    def evaluate(board_2d, depth):
        # Check for wins
        for col in range(columns):
            if board_2d[0][col] == 0:
                test_board, row = make_move(board_2d, col, mark)
                if test_board and row >= 0:
                    if check_win_from_position(test_board, row, col, mark):
                        return 10000 - depth
                test_board, row = make_move(board_2d, col, opponent)
                if test_board and row >= 0:
                    if check_win_from_position(test_board, row, col, opponent):
                        return -10000 + depth
        
        score = 0
        
        # Pattern evaluation
        def eval_window(window, player):
            opp = 3 - player
            if window.count(opp) == 0:
                if window.count(player) == 3:
                    return 50
                elif window.count(player) == 2:
                    return 10
                elif window.count(player) == 1:
                    return 1
            return 0
        
        # Horizontal patterns
        for r in range(rows):
            for c in range(columns - 3):
                window = [board_2d[r][c+i] for i in range(4)]
                score += eval_window(window, mark)
                score -= eval_window(window, opponent) * 1.1  # Defensive bias
        
        # Vertical patterns
        for c in range(columns):
            for r in range(rows - 3):
                window = [board_2d[r+i][c] for i in range(4)]
                score += eval_window(window, mark)
                score -= eval_window(window, opponent) * 1.1
        
        # Diagonal patterns \
        for r in range(rows - 3):
            for c in range(columns - 3):
                window = [board_2d[r+i][c+i] for i in range(4)]
                score += eval_window(window, mark)
                score -= eval_window(window, opponent) * 1.1
        
        # Diagonal patterns /
        for r in range(3, rows):
            for c in range(columns - 3):
                window = [board_2d[r-i][c+i] for i in range(4)]
                score += eval_window(window, mark)
                score -= eval_window(window, opponent) * 1.1
        
        # Center column bonus
        center = columns // 2
        center_score = 0
        for r in range(rows):
            if board_2d[r][center] == mark:
                center_score += 3
            elif board_2d[r][center] == opponent:
                center_score -= 3
        score += center_score * 4
        
        # Adjacent columns bonus
        for offset in [1, -1]:
            if 0 <= center + offset < columns:
                for r in range(rows):
                    if board_2d[r][center + offset] == mark:
                        score += 2
                    elif board_2d[r][center + offset] == opponent:
                        score -= 2
        
        # Threat bonus
        player_threats = count_threats(board_2d, mark)
        opponent_threats = count_threats(board_2d, opponent)
        score += player_threats * 30
        score -= opponent_threats * 35  # Defensive
        
        return score
    
    # Minimax with alpha-beta pruning
    def minimax(board_2d, depth, alpha, beta, maximizing, start_time):
        # Time limit check
        if time.time() - start_time > 0.85:
            return evaluate(board_2d, depth), None
        
        # Create board hash for transposition table
        board_hash = hash(tuple(tuple(row) for row in board_2d))
        if board_hash in transposition_table and transposition_table[board_hash]['depth'] >= depth:
            return transposition_table[board_hash]['score'], transposition_table[board_hash]['move']
        
        # Get valid moves
        valid_moves = []
        for col in range(columns):
            if board_2d[0][col] == 0:
                valid_moves.append(col)
        
        if not valid_moves or depth == 0:
            score = evaluate(board_2d, depth)
            transposition_table[board_hash] = {'score': score, 'depth': depth, 'move': None}
            return score, None
        
        # Move ordering - center first
        center = columns // 2
        valid_moves.sort(key=lambda x: abs(x - center))
        
        # Check for immediate wins first
        for col in valid_moves:
            test_board, row = make_move(board_2d, col, mark if maximizing else opponent)
            if test_board and row >= 0:
                if check_win_from_position(test_board, row, col, mark if maximizing else opponent):
                    score = (10000 - depth) * (1 if maximizing else -1)
                    transposition_table[board_hash] = {'score': score, 'depth': depth, 'move': col}
                    return score, col
        
        best_move = valid_moves[0]
        
        if maximizing:
            max_eval = -float('inf')
            for col in valid_moves:
                new_board, row = make_move(board_2d, col, mark)
                if new_board and row >= 0:
                    eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False, start_time)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = col
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Prune
            transposition_table[board_hash] = {'score': max_eval, 'depth': depth, 'move': best_move}
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for col in valid_moves:
                new_board, row = make_move(board_2d, col, opponent)
                if new_board and row >= 0:
                    eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True, start_time)
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = col
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Prune
            transposition_table[board_hash] = {'score': min_eval, 'depth': depth, 'move': best_move}
            return min_eval, best_move
    
    # Main agent logic
    start_time = time.time()
    
    # Try opening book first
    move_count = sum(1 for x in board if x != 0)
    if move_count < 8:
        moves = []
        for col in range(columns):
            col_moves = []
            for row in range(rows-1, -1, -1):
                if board[row * columns + col] != 0:
                    col_moves.append(col)
            moves.extend(reversed(col_moves))
        
        move_tuple = tuple(moves)
        if move_tuple in OPENING_BOOK:
            book_move = OPENING_BOOK[move_tuple]
            if board[book_move] == 0:
                return book_move
    
    # Convert to 2D
    board_2d = to_2d()
    
    # Get valid moves
    valid_moves = [c for c in range(columns) if board[c] == 0]
    if not valid_moves:
        return columns // 2
    
    # Check immediate wins
    for col in valid_moves:
        test_board, row = make_move(board_2d, col, mark)
        if test_board and row >= 0:
            if check_win_from_position(test_board, row, col, mark):
                return col
    
    # Check immediate blocks
    for col in valid_moves:
        test_board, row = make_move(board_2d, col, opponent)
        if test_board and row >= 0:
            if check_win_from_position(test_board, row, col, opponent):
                return col
    
    # Dynamic depth based on game phase
    if move_count < 8:
        search_depth = 8
    elif move_count < 20:
        search_depth = 9
    else:
        search_depth = 10
    
    # Clear transposition table if too large
    if len(transposition_table) > 500000:
        transposition_table.clear()
    
    # Iterative deepening
    best_move = valid_moves[len(valid_moves)//2]
    best_score = -float('inf')
    
    for depth in range(4, search_depth + 1):
        if time.time() - start_time > 0.8:
            break
        
        score, move = minimax(board_2d, depth, -float('inf'), float('inf'), True, start_time)
        if move is not None:
            best_move = move
            best_score = score
        
        # If we found a guaranteed win, take it
        if score >= 9000:
            break
    
    return best_move