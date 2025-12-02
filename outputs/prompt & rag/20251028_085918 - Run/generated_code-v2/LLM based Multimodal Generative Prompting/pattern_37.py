import svgwrite

def generate_svg(thought_prompt: str, problem_context: str) -> str:
    """
    Generates a simple SVG image based on a thought prompt and problem context.
    This is a simplified illustration, not a full-fledged AI image generator.
    """
    dwg = svgwrite.Drawing(size=('400px', '300px'))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill='white'))

    # Add a title based on the thought prompt
    dwg.add(dwg.text(thought_prompt, insert=(20, 30), fill='black', font_size='18px', font_weight='bold'))

    if "initial conditions" in thought_prompt.lower():
        # Example: Draw a simple object on a surface
        dwg.add(dwg.rect(insert=(50, 200), size=(100, 50), fill='lightblue', stroke='blue', stroke_width=2))
        dwg.add(dwg.line(start=(0, 250), end=(400, 250), stroke='gray', stroke_width=3))
        dwg.add(dwg.text("Object on a surface", insert=(50, 190), fill='black', font_size='12px'))
    elif "forces involved" in thought_prompt.lower() or "free-body diagram" in thought_prompt.lower():
        # Example: Draw a free-body diagram
        dwg.add(dwg.rect(insert=(150, 100), size=(100, 100), fill='lightcoral', stroke='red', stroke_width=2))
        dwg.add(dwg.line(start=(200, 200), end=(200, 250), stroke='black', stroke_width=2, marker_end=dwg.marker(orient='auto', size=(5, 5), refX=0, refY=2.5).add(dwg.path(d='M0,0 L0,5 L5,2.5 z'))))
        dwg.add(dwg.text("Gravity (Fg)", insert=(205, 265), fill='black', font_size='12px'))
        dwg.add(dwg.line(start=(200, 100), end=(200, 50), stroke='black', stroke_width=2, marker_end=dwg.marker(orient='auto', size=(5, 5), refX=0, refY=2.5).add(dwg.path(d='M0,0 L0,5 L5,2.5 z'))))
        dwg.add(dwg.text("Normal (Fn)", insert=(205, 45), fill='black', font_size='12px'))
        dwg.add(dwg.text("Free-Body Diagram", insert=(150, 90), fill='black', font_size='12px'))
    elif "vector decomposition" in thought_prompt.lower():
        # Example: Vector decomposition
        dwg.add(dwg.line(start=(50, 200), end=(200, 100), stroke='green', stroke_width=2, marker_end=dwg.marker(orient='auto', size=(5, 5), refX=0, refY=2.5).add(dwg.path(d='M0,0 L0,5 L5,2.5 z'))))
        dwg.add(dwg.text("Original Vector", insert=(60, 190), fill='green', font_size='12px'))
        dwg.add(dwg.line(start=(50, 200), end=(200, 200), stroke='blue', stroke_width=1, stroke_dasharray='5,5'))
        dwg.add(dwg.text("X-component", insert=(100, 215), fill='blue', font_size='12px'))
        dwg.add(dwg.line(start=(200, 200), end=(200, 100), stroke='red', stroke_width=1, stroke_dasharray='5,5'))
        dwg.add(dwg.text("Y-component", insert=(205, 150), fill='red', font_size='12px'))
        dwg.add(dwg.text("Vector Decomposition", insert=(50, 50), fill='black', font_size='12px'))
    else:
        # Generic drawing based on problem context
        dwg.add(dwg.text(f"Visualizing: {problem_context}", insert=(20, 150), fill='darkgray', font_size='14px'))
        dwg.add(dwg.rect(insert=(100, 100), size=(200, 100), fill='lightgray', stroke='black', stroke_width=1, rx=10, ry=10))

    return dwg.to_string()