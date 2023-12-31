from constants import *

def get_move_from_index(index: int) -> tuple[tuple[int, int], Move]:

    slide, index = get_slide_from_index(index)

    from_pos = get_pos_from_index(index)

    return from_pos, slide
    
def get_index_from_move(action: tuple[tuple[int, int], Move]) -> int:
    '''Returns the index corresponding to the given move'''
    from_pos, slide = action
    index = get_index_from_slide(from_pos, slide)
    # print("Offset: ", from_pos, index)
    # print("Index: ", from_pos, get_index_from_pos(from_pos))

    index += get_index_from_pos(from_pos)

    return index 

def get_slide_from_index(index: int) -> tuple[Move, int]:
    slide_number = index // (ACTION_SPACE//4)
    index -= slide_number * (ACTION_SPACE//4)

    if slide_number == 0:
        if not INVALID_MOVES:
            index += 1 + max(0, min(index + 2, ACTION_SPACE//4 - (N - 1) + 1 + 1) - N)
            # for _ in range(N, min(index + 1, ACTION_SPACE//4 - (N - 1) + 1 + 1)):
            #     index += 1
        slide = Move.TOP
    elif slide_number == 1:
        if not INVALID_MOVES:
            index +=  max(0, min(index + 1, ACTION_SPACE//4 - (N - 1) + 1) - (N - 1))
            # for _ in range(N - 1, min(index + 1, ACTION_SPACE//4 - (N - 1) + 1)):
            #     index += 1
        slide = Move.BOTTOM
    elif slide_number == 2:
        if not INVALID_MOVES:
            index += N
        slide = Move.LEFT
    else:
        slide = Move.RIGHT

    return slide, index

def get_pos_from_index(index) -> tuple[int, int]:
    if index <= N - 1:
        from_pos = (0, index)
    elif index >= 3*(N - 1) - 1:
        index -= 3*(N - 1) - 1
        from_pos = (N - 1, index)
    else:
        index -= N 
        from_pos = (1 + index // 2, index % 2 * (N - 1))

    return from_pos

def get_index_from_slide(from_pos: tuple[int, int], slide: Move) -> int:
    index = 0
    if slide == Move.TOP:
        index += 0
        if not INVALID_MOVES:
            index -= from_pos[0] + 1
    elif slide == Move.BOTTOM:
        index += ACTION_SPACE//4
        if not INVALID_MOVES:
            index -= from_pos[0]
    elif slide == Move.LEFT:
        index += 2*ACTION_SPACE//4
        if not INVALID_MOVES:
            index -= N
    else:
        index += 3*ACTION_SPACE//4

    return index

def get_index_from_pos(from_pos: tuple[int, int]) -> int:
    index = 0
    if from_pos[0] == 0:
        index += from_pos[1]
    elif from_pos[0] == N - 1:
        index += 3*(N - 1) - 1 + from_pos[1]
    else:
        index += N + 2*(from_pos[0] - 1) + from_pos[1] // (N - 1)

    return index




        


