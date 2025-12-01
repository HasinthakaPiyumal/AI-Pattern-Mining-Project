import gradio as gr

def _generate_svg_circle(cx, cy, r, color="blue", stroke="black", stroke_width="2"):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}" />'

def _generate_svg_rectangle(x, y, width, height, color="green", stroke="black", stroke_width="2"):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}" />'

def _generate_svg_line(x1, y1, x2, y2, color="black", stroke_width="2"):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke_width}" />'

def _generate_svg_text(x, y, text, font_size="16", color="black"):
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{font_size}" fill="{color}">{text}</text>'

def _wrap_svg_content(content, width=400, height=300):
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{content}</svg>'

def _simulate_geometry_problem(problem_description):
    steps = []

    # Step 1: Initial Problem Setup - A triangle
    svg_content_1 = _generate_svg_line(50, 250, 200, 50) + \
                    _generate_svg_line(200, 50, 350, 250) + \
                    _generate_svg_line(350, 250, 50, 250) + \
                    _generate_svg_text(40, 265, "A") + \
                    _generate_svg_text(195, 40, "B") + \
                    _generate_svg_text(355, 265, "C")
    steps.append((_wrap_svg_content(svg_content_1), "Let's consider a triangle ABC."))

    # Step 2: Add an altitude
    svg_content_2 = svg_content_1 + \
                    _generate_svg_line(200, 50, 200, 250, color="red") + \
                    _generate_svg_text(185, 265, "D")
    steps.append((_wrap_svg_content(svg_content_2), "Draw an altitude from vertex B to side AC, intersecting at D."))

    # Step 3: Highlight a right angle
    svg_content_3 = svg_content_2 + \
                    _generate_svg_rectangle(190, 230, 20, 20, color="none", stroke="red")
    steps.append((_wrap_svg_content(svg_content_3), "Notice the right angle formed at D, creating two right triangles ABD and BCD."))

    return steps

def _simulate_physics_problem(problem_description):
    steps = []

    # Step 1: Object on a surface
    svg_content_1 = _generate_svg_rectangle(150, 200, 100, 50, color="lightgray") + \
                    _generate_svg_line(50, 250, 350, 250, color="brown", stroke_width="5") + \
                    _generate_svg_text(180, 230, "Block")
    steps.append((_wrap_svg_content(svg_content_1), "A block rests on a horizontal surface."))

    # Step 2: Apply a force
    svg_content_2 = svg_content_1 + \
                    _generate_svg_line(250, 225, 300, 225, color="red", stroke_width="3") + \
                    _generate_svg_line(285, 220, 300, 225, color="red", stroke_width="3") + \
                    _generate_svg_line(285, 230, 300, 225, color="red", stroke_width="3") + \
                    _generate_svg_text(260, 215, "F")
    steps.append((_wrap_svg_content(svg_content_2), "A force F is applied horizontally to the block."))

    # Step 3: Show friction force
    svg_content_3 = svg_content_2 + \
                    _generate_svg_line(150, 225, 100, 225, color="blue", stroke_width="3") + \
                    _generate_svg_line(115, 220, 100, 225, color="blue", stroke_width="3") + \
                    _generate_svg_line(115, 230, 100, 225, color="blue", stroke_width="3") + \
                    _generate_svg_text(105, 215, "f_k")
    steps.append((_wrap_svg_content(svg_content_3), "Due to the force, kinetic friction (f_k) acts opposite to the motion."))

    return steps

def solve_stem_problem(problem_text):
    problem_text_lower = problem_text.lower()
    if "geometry" in problem_text_lower or "triangle" in problem_text_lower or "angles" in problem_text_lower:
        return _simulate_geometry_problem(problem_text)
    elif "physics" in problem_text_lower or "force" in problem_text_lower or "block" in problem_text_lower:
        return _simulate_physics_problem(problem_text)
    else:
        return [(_wrap_svg_content(_generate_svg_text(50, 150, "Problem type not recognized or simulated.", 20, "red")), "Please try a geometry or physics problem to see an example Chain of Images.")]

def create_interface():
    problem_input = gr.Textbox(lines=5, label="Enter your STEM problem (e.g., 'Explain the area of a triangle' or 'Describe a block on an incline').")
    
    # Dynamic output components based on the number of steps
    outputs = []
    for i in range(5):  # Max 5 steps for demonstration
        outputs.append(gr.HTML(label=f"Visual Step {i+1}"))
        outputs.append(gr.Textbox(label=f"Explanation {i+1}", lines=2, interactive=False))

    def process_problem(problem_text):
        results = solve_stem_problem(problem_text)
        output_values = []
        for i in range(len(outputs) // 2):
            if i < len(results):
                svg_html, explanation = results[i]
                output_values.append(svg_html)
                output_values.append(explanation)
            else:
                output_values.append("") # Empty SVG
                output_values.append("") # Empty explanation
        return tuple(output_values)

    demo = gr.Interface(
        fn=process_problem,
        inputs=problem_input,
        outputs=outputs,
        title="Interactive STEM Problem Solver (Chain of Images)",
        description="Enter a STEM problem and see a step-by-step visual explanation."
    )
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()