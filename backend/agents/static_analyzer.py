import ast
from dataclasses import dataclass

@dataclass
class Issue:
    line: int
    severity: str     
    code: str          
    message: str       

def analyze(source_code: str) -> list[Issue]:
    """
    Parse a PyTorch source file and return a list of detected issues.
    source_code: raw Python source code as a string
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return [Issue(line=e.lineno or 0, severity="error",
                      code="SYNTAX_ERROR", message=str(e))]

    issues = []
    checker = PyTorchChecker(issues)
    checker.visit(tree)
    return issues


class PyTorchChecker(ast.NodeVisitor):

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        self._in_eval_block = False   
    def _is_method_call(self, node, attr: str) -> bool:
        return (
            isinstance(node, ast.Call) and
            isinstance(node.func, ast.Attribute) and
            node.func.attr == attr
        )

  
    def visit_For(self, node: ast.For):
        for child in ast.walk(node):
            if self._is_method_call(child, "item"):
                self.issues.append(Issue(
                    line=child.lineno,
                    severity="warning",
                    code="ITEM_IN_LOOP",
                    message=(
                        ".item() inside a loop forces a GPU-CPU sync on every "
                        "iteration. Accumulate the tensor and call .item() once "
                        "outside the loop."
                    )
                ))
        self.generic_visit(node)

    
    def visit_Module(self, node: ast.Module):
        source = ast.dump(node)
        has_eval   = "eval" in source
        has_no_grad = "no_grad" in source

        if has_eval and not has_no_grad:
            self.issues.append(Issue(
                line=0,
                severity="warning",
                code="NO_GRAD_MISSING",
                message=(
                    "model.eval() detected but torch.no_grad() context manager "
                    "is missing. Wrap your inference loop with "
                    "`with torch.no_grad():` to disable gradient tracking and "
                    "save memory."
                )
            ))
        self.generic_visit(node)

   
    def visit_Call(self, node: ast.Call):
        TENSOR_CONSTRUCTORS = {"zeros", "ones", "tensor", "empty", "rand", "randn"}

        if (
            isinstance(node.func, ast.Attribute) and
            node.func.attr in TENSOR_CONSTRUCTORS and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == "torch"
        ):
            
            has_device = any(kw.arg == "device" for kw in node.keywords)
            if not has_device:
                self.issues.append(Issue(
                    line=node.lineno,
                    severity="warning",
                    code="TENSOR_NO_DEVICE",
                    message=(
                        f"torch.{node.func.attr}() called without a `device` "
                        "argument. Explicitly pass device=device to avoid "
                        "unintended CPU tensors when your model is on GPU."
                    )
                ))
        self.generic_visit(node)

  
    def visit_Call_dataloader(self, node: ast.Call):
      
        has_workers = any(kw.arg == "num_workers" for kw in node.keywords)
        if not has_workers:
            self.issues.append(Issue(
                line=node.lineno,
                severity="warning",
                code="DATALOADER_NO_WORKERS",
                message=(
                    "DataLoader created without num_workers. Set num_workers >= 2 "
                    "to enable parallel data loading and avoid CPU bottlenecks."
                )
            ))

    def visit_Call(self, node: ast.Call):  
        TENSOR_CONSTRUCTORS = {"zeros", "ones", "tensor", "empty", "rand", "randn"}

       
        if (
            isinstance(node.func, ast.Attribute) and
            node.func.attr in TENSOR_CONSTRUCTORS and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == "torch"
        ):
            has_device = any(kw.arg == "device" for kw in node.keywords)
            if not has_device:
                self.issues.append(Issue(
                    line=node.lineno,
                    severity="warning",
                    code="TENSOR_NO_DEVICE",
                    message=(
                        f"torch.{node.func.attr}() called without a `device` "
                        "argument. Explicitly pass device=device to avoid "
                        "unintended CPU tensors when your model is on GPU."
                    )
                ))

       
        if (
            isinstance(node.func, ast.Name) and node.func.id == "DataLoader"
        ) or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "DataLoader"
        ):
            self.visit_Call_dataloader(node)

        self.generic_visit(node)