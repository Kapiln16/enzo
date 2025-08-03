def solve(expression: str) -> str:
    try:
        return str(eval(expression))
    except:
        return "Sorry, I couldn't compute that."
