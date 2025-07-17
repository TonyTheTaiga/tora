import SwiftUI

struct ToraIcon: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        let width = rect.size.width
        let height = rect.size.height
        path.move(to: CGPoint(x: 0.95126 * width, y: 0.39995 * height))
        path.addLine(to: CGPoint(x: 0.95126 * width, y: 0.29084 * height))
        path.addCurve(
            to: CGPoint(x: 0.94807 * width, y: 0.27273 * height),
            control1: CGPoint(x: 0.95126 * width, y: 0.28334 * height),
            control2: CGPoint(x: 0.95003 * width, y: 0.27648 * height))
        path.addLine(to: CGPoint(x: 0.91917 * width, y: 0.21813 * height))
        path.addCurve(
            to: CGPoint(x: 0.91276 * width, y: 0.21813 * height),
            control1: CGPoint(x: 0.91718 * width, y: 0.21438 * height),
            control2: CGPoint(x: 0.91475 * width, y: 0.21438 * height))
        path.addLine(to: CGPoint(x: 0.88386 * width, y: 0.27273 * height))
        path.addCurve(
            to: CGPoint(x: 0.88067 * width, y: 0.29084 * height),
            control1: CGPoint(x: 0.88187 * width, y: 0.27648 * height),
            control2: CGPoint(x: 0.88067 * width, y: 0.28334 * height))
        path.addLine(to: CGPoint(x: 0.88067 * width, y: 0.39995 * height))
        path.addCurve(
            to: CGPoint(x: 0.88386 * width, y: 0.41805 * height),
            control1: CGPoint(x: 0.88067 * width, y: 0.40744 * height),
            control2: CGPoint(x: 0.8819 * width, y: 0.4143 * height))
        path.addLine(to: CGPoint(x: 0.91276 * width, y: 0.47265 * height))
        path.addCurve(
            to: CGPoint(x: 0.91917 * width, y: 0.47265 * height),
            control1: CGPoint(x: 0.91475 * width, y: 0.4764 * height),
            control2: CGPoint(x: 0.91718 * width, y: 0.4764 * height))
        path.addLine(to: CGPoint(x: 0.94807 * width, y: 0.41805 * height))
        path.addCurve(
            to: CGPoint(x: 0.95126 * width, y: 0.39995 * height),
            control1: CGPoint(x: 0.95006 * width, y: 0.4143 * height),
            control2: CGPoint(x: 0.95126 * width, y: 0.40744 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.85658 * width, y: 0.19416 * height))
        path.addLine(to: CGPoint(x: 0.85658 * width, y: 0.23011 * height))
        path.addCurve(
            to: CGPoint(x: 0.85868 * width, y: 0.24291 * height),
            control1: CGPoint(x: 0.85658 * width, y: 0.23523 * height),
            control2: CGPoint(x: 0.85736 * width, y: 0.24008 * height))
        path.addLine(to: CGPoint(x: 0.86584 * width, y: 0.25873 * height))
        path.addCurve(
            to: CGPoint(x: 0.87113 * width, y: 0.25873 * height),
            control1: CGPoint(x: 0.86744 * width, y: 0.26221 * height),
            control2: CGPoint(x: 0.86953 * width, y: 0.26221 * height))
        path.addLine(to: CGPoint(x: 0.87829 * width, y: 0.24291 * height))
        path.addCurve(
            to: CGPoint(x: 0.88039 * width, y: 0.23011 * height),
            control1: CGPoint(x: 0.87961 * width, y: 0.24008 * height),
            control2: CGPoint(x: 0.88039 * width, y: 0.23523 * height))
        path.addLine(to: CGPoint(x: 0.88039 * width, y: 0.19416 * height))
        path.addCurve(
            to: CGPoint(x: 0.87829 * width, y: 0.18136 * height),
            control1: CGPoint(x: 0.88039 * width, y: 0.18904 * height),
            control2: CGPoint(x: 0.87961 * width, y: 0.1842 * height))
        path.addLine(to: CGPoint(x: 0.87113 * width, y: 0.16554 * height))
        path.addCurve(
            to: CGPoint(x: 0.86584 * width, y: 0.16554 * height),
            control1: CGPoint(x: 0.86953 * width, y: 0.16206 * height),
            control2: CGPoint(x: 0.86744 * width, y: 0.16206 * height))
        path.addLine(to: CGPoint(x: 0.85868 * width, y: 0.18136 * height))
        path.addCurve(
            to: CGPoint(x: 0.85658 * width, y: 0.19416 * height),
            control1: CGPoint(x: 0.85736 * width, y: 0.1842 * height),
            control2: CGPoint(x: 0.85658 * width, y: 0.18904 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.94967 * width, y: 0.19416 * height))
        path.addLine(to: CGPoint(x: 0.94967 * width, y: 0.23011 * height))
        path.addCurve(
            to: CGPoint(x: 0.95176 * width, y: 0.24291 * height),
            control1: CGPoint(x: 0.94967 * width, y: 0.23523 * height),
            control2: CGPoint(x: 0.95045 * width, y: 0.24008 * height))
        path.addLine(to: CGPoint(x: 0.95895 * width, y: 0.25873 * height))
        path.addCurve(
            to: CGPoint(x: 0.96424 * width, y: 0.25873 * height),
            control1: CGPoint(x: 0.96055 * width, y: 0.26221 * height),
            control2: CGPoint(x: 0.96265 * width, y: 0.26221 * height))
        path.addLine(to: CGPoint(x: 0.97143 * width, y: 0.24291 * height))
        path.addCurve(
            to: CGPoint(x: 0.97353 * width, y: 0.23011 * height),
            control1: CGPoint(x: 0.97275 * width, y: 0.24008 * height),
            control2: CGPoint(x: 0.97353 * width, y: 0.23523 * height))
        path.addLine(to: CGPoint(x: 0.97353 * width, y: 0.19416 * height))
        path.addCurve(
            to: CGPoint(x: 0.97143 * width, y: 0.18136 * height),
            control1: CGPoint(x: 0.97353 * width, y: 0.18904 * height),
            control2: CGPoint(x: 0.97275 * width, y: 0.1842 * height))
        path.addLine(to: CGPoint(x: 0.96424 * width, y: 0.16554 * height))
        path.addCurve(
            to: CGPoint(x: 0.95895 * width, y: 0.16554 * height),
            control1: CGPoint(x: 0.96265 * width, y: 0.16206 * height),
            control2: CGPoint(x: 0.96055 * width, y: 0.16206 * height))
        path.addLine(to: CGPoint(x: 0.95176 * width, y: 0.18136 * height))
        path.addCurve(
            to: CGPoint(x: 0.94967 * width, y: 0.19416 * height),
            control1: CGPoint(x: 0.95045 * width, y: 0.1842 * height),
            control2: CGPoint(x: 0.94967 * width, y: 0.18904 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.88621 * width, y: 0.13463 * height))
        path.addLine(to: CGPoint(x: 0.88621 * width, y: 0.17057 * height))
        path.addCurve(
            to: CGPoint(x: 0.88831 * width, y: 0.18346 * height),
            control1: CGPoint(x: 0.88621 * width, y: 0.17569 * height),
            control2: CGPoint(x: 0.88699 * width, y: 0.18054 * height))
        path.addLine(to: CGPoint(x: 0.89547 * width, y: 0.1992 * height))
        path.addCurve(
            to: CGPoint(x: 0.90076 * width, y: 0.1992 * height),
            control1: CGPoint(x: 0.89706 * width, y: 0.20276 * height),
            control2: CGPoint(x: 0.89916 * width, y: 0.20276 * height))
        path.addLine(to: CGPoint(x: 0.90792 * width, y: 0.18346 * height))
        path.addCurve(
            to: CGPoint(x: 0.91002 * width, y: 0.17057 * height),
            control1: CGPoint(x: 0.90924 * width, y: 0.18063 * height),
            control2: CGPoint(x: 0.91002 * width, y: 0.17578 * height))
        path.addLine(to: CGPoint(x: 0.91002 * width, y: 0.13463 * height))
        path.addCurve(
            to: CGPoint(x: 0.90792 * width, y: 0.12173 * height),
            control1: CGPoint(x: 0.91002 * width, y: 0.1295 * height),
            control2: CGPoint(x: 0.90924 * width, y: 0.12466 * height))
        path.addLine(to: CGPoint(x: 0.90076 * width, y: 0.106 * height))
        path.addCurve(
            to: CGPoint(x: 0.89547 * width, y: 0.106 * height),
            control1: CGPoint(x: 0.89916 * width, y: 0.10243 * height),
            control2: CGPoint(x: 0.89706 * width, y: 0.10243 * height))
        path.addLine(to: CGPoint(x: 0.88831 * width, y: 0.12173 * height))
        path.addCurve(
            to: CGPoint(x: 0.88621 * width, y: 0.13463 * height),
            control1: CGPoint(x: 0.88699 * width, y: 0.12457 * height),
            control2: CGPoint(x: 0.88621 * width, y: 0.12941 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.91844 * width, y: 0.13463 * height))
        path.addLine(to: CGPoint(x: 0.91844 * width, y: 0.17057 * height))
        path.addCurve(
            to: CGPoint(x: 0.92054 * width, y: 0.18337 * height),
            control1: CGPoint(x: 0.91844 * width, y: 0.17569 * height),
            control2: CGPoint(x: 0.91922 * width, y: 0.18054 * height))
        path.addLine(to: CGPoint(x: 0.9277 * width, y: 0.1991 * height))
        path.addCurve(
            to: CGPoint(x: 0.93299 * width, y: 0.1991 * height),
            control1: CGPoint(x: 0.9293 * width, y: 0.20267 * height),
            control2: CGPoint(x: 0.9314 * width, y: 0.20267 * height))
        path.addLine(to: CGPoint(x: 0.94015 * width, y: 0.18337 * height))
        path.addCurve(
            to: CGPoint(x: 0.94225 * width, y: 0.17057 * height),
            control1: CGPoint(x: 0.94147 * width, y: 0.18054 * height),
            control2: CGPoint(x: 0.94225 * width, y: 0.17569 * height))
        path.addLine(to: CGPoint(x: 0.94225 * width, y: 0.13463 * height))
        path.addCurve(
            to: CGPoint(x: 0.94015 * width, y: 0.12182 * height),
            control1: CGPoint(x: 0.94225 * width, y: 0.1295 * height),
            control2: CGPoint(x: 0.94147 * width, y: 0.12466 * height))
        path.addLine(to: CGPoint(x: 0.93299 * width, y: 0.10609 * height))
        path.addCurve(
            to: CGPoint(x: 0.9277 * width, y: 0.10609 * height),
            control1: CGPoint(x: 0.9314 * width, y: 0.10252 * height),
            control2: CGPoint(x: 0.9293 * width, y: 0.10252 * height))
        path.addLine(to: CGPoint(x: 0.92054 * width, y: 0.12182 * height))
        path.addCurve(
            to: CGPoint(x: 0.91844 * width, y: 0.13463 * height),
            control1: CGPoint(x: 0.91922 * width, y: 0.12466 * height),
            control2: CGPoint(x: 0.91844 * width, y: 0.1295 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.91422 * width, y: 0.56082 * height))
        path.addCurve(
            to: CGPoint(x: 0.82843 * width, y: 0.28041 * height),
            control1: CGPoint(x: 0.8669 * width, y: 0.56082 * height),
            control2: CGPoint(x: 0.82843 * width, y: 0.43506 * height))
        path.addCurve(
            to: CGPoint(x: 0.91422 * width, y: 0),
            control1: CGPoint(x: 0.82843 * width, y: 0.12575 * height),
            control2: CGPoint(x: 0.8669 * width, y: 0))
        path.addCurve(
            to: CGPoint(x: width, y: 0.28041 * height), control1: CGPoint(x: 0.96153 * width, y: 0),
            control2: CGPoint(x: width, y: 0.12575 * height))
        path.addCurve(
            to: CGPoint(x: 0.91422 * width, y: 0.56082 * height),
            control1: CGPoint(x: width, y: 0.43506 * height),
            control2: CGPoint(x: 0.96153 * width, y: 0.56082 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.91422 * width, y: 0.03475 * height))
        path.addCurve(
            to: CGPoint(x: 0.83906 * width, y: 0.28041 * height),
            control1: CGPoint(x: 0.87278 * width, y: 0.03475 * height),
            control2: CGPoint(x: 0.83906 * width, y: 0.14496 * height))
        path.addCurve(
            to: CGPoint(x: 0.91422 * width, y: 0.52607 * height),
            control1: CGPoint(x: 0.83906 * width, y: 0.41586 * height),
            control2: CGPoint(x: 0.87278 * width, y: 0.52607 * height))
        path.addCurve(
            to: CGPoint(x: 0.98937 * width, y: 0.28041 * height),
            control1: CGPoint(x: 0.95565 * width, y: 0.52607 * height),
            control2: CGPoint(x: 0.98937 * width, y: 0.41586 * height))
        path.addCurve(
            to: CGPoint(x: 0.91422 * width, y: 0.03475 * height),
            control1: CGPoint(x: 0.98937 * width, y: 0.14496 * height),
            control2: CGPoint(x: 0.95565 * width, y: 0.03475 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.10548 * width, y: 0.64588 * height))
        path.addCurve(
            to: CGPoint(x: 0.11264 * width, y: 0.62246 * height),
            control1: CGPoint(x: 0.10548 * width, y: 0.63188 * height),
            control2: CGPoint(x: 0.10833 * width, y: 0.62246 * height))
        path.addLine(to: CGPoint(x: 0.19308 * width, y: 0.62246 * height))
        path.addCurve(
            to: CGPoint(x: 0.20025 * width, y: 0.64588 * height),
            control1: CGPoint(x: 0.19736 * width, y: 0.62246 * height),
            control2: CGPoint(x: 0.20025 * width, y: 0.63179 * height))
        path.addLine(to: CGPoint(x: 0.20025 * width, y: 0.89482 * height))
        path.addCurve(
            to: CGPoint(x: 0.16807 * width, y: height),
            control1: CGPoint(x: 0.20025 * width, y: 0.95793 * height),
            control2: CGPoint(x: 0.18774 * width, y: height))
        path.addLine(to: CGPoint(x: 0.03218 * width, y: height))
        path.addCurve(
            to: CGPoint(x: 0, y: 0.89482 * height), control1: CGPoint(x: 0.01251 * width, y: height),
            control2: CGPoint(x: 0, y: 0.95793 * height))
        path.addLine(to: CGPoint(x: 0, y: 0.35367 * height))
        path.addCurve(
            to: CGPoint(x: 0.00716 * width, y: 0.33025 * height),
            control1: CGPoint(x: 0, y: 0.33967 * height),
            control2: CGPoint(x: 0.00285 * width, y: 0.33025 * height))
        path.addLine(to: CGPoint(x: 0.07689 * width, y: 0.33025 * height))
        path.addCurve(
            to: CGPoint(x: 0.08405 * width, y: 0.35367 * height),
            control1: CGPoint(x: 0.08117 * width, y: 0.33025 * height),
            control2: CGPoint(x: 0.08405 * width, y: 0.33958 * height))
        path.addLine(to: CGPoint(x: 0.08405 * width, y: 0.439 * height))
        path.addLine(to: CGPoint(x: 0.19311 * width, y: 0.439 * height))
        path.addCurve(
            to: CGPoint(x: 0.20027 * width, y: 0.46241 * height),
            control1: CGPoint(x: 0.19739 * width, y: 0.439 * height),
            control2: CGPoint(x: 0.20027 * width, y: 0.44833 * height))
        path.addLine(to: CGPoint(x: 0.20027 * width, y: 0.56997 * height))
        path.addCurve(
            to: CGPoint(x: 0.19311 * width, y: 0.59338 * height),
            control1: CGPoint(x: 0.20027 * width, y: 0.58396 * height),
            control2: CGPoint(x: 0.19742 * width, y: 0.59338 * height))
        path.addLine(to: CGPoint(x: 0.09479 * width, y: 0.59338 * height))
        path.addLine(to: CGPoint(x: 0.09479 * width, y: 0.91833 * height))
        path.addLine(to: CGPoint(x: 0.10551 * width, y: 0.91833 * height))
        path.addLine(to: CGPoint(x: 0.10551 * width, y: 0.64597 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.41118 * width, y: 0.89482 * height))
        path.addCurve(
            to: CGPoint(x: 0.379 * width, y: height),
            control1: CGPoint(x: 0.41118 * width, y: 0.95793 * height),
            control2: CGPoint(x: 0.39867 * width, y: height))
        path.addLine(to: CGPoint(x: 0.24314 * width, y: height))
        path.addCurve(
            to: CGPoint(x: 0.21096 * width, y: 0.89482 * height),
            control1: CGPoint(x: 0.22347 * width, y: height),
            control2: CGPoint(x: 0.21096 * width, y: 0.95793 * height))
        path.addLine(to: CGPoint(x: 0.21096 * width, y: 0.54417 * height))
        path.addCurve(
            to: CGPoint(x: 0.24314 * width, y: 0.439 * height),
            control1: CGPoint(x: 0.21096 * width, y: 0.48107 * height),
            control2: CGPoint(x: 0.22347 * width, y: 0.439 * height))
        path.addLine(to: CGPoint(x: 0.379 * width, y: 0.439 * height))
        path.addCurve(
            to: CGPoint(x: 0.41118 * width, y: 0.54417 * height),
            control1: CGPoint(x: 0.39867 * width, y: 0.439 * height),
            control2: CGPoint(x: 0.41118 * width, y: 0.48107 * height))
        path.addLine(to: CGPoint(x: 0.41118 * width, y: 0.89482 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.31644 * width, y: 0.52076 * height))
        path.addLine(to: CGPoint(x: 0.30573 * width, y: 0.52076 * height))
        path.addLine(to: CGPoint(x: 0.30573 * width, y: 0.91815 * height))
        path.addLine(to: CGPoint(x: 0.31644 * width, y: 0.91815 * height))
        path.addLine(to: CGPoint(x: 0.31644 * width, y: 0.52076 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.51666 * width, y: 0.52076 * height))
        path.addLine(to: CGPoint(x: 0.51666 * width, y: 0.97659 * height))
        path.addCurve(
            to: CGPoint(x: 0.5095 * width, y: height),
            control1: CGPoint(x: 0.51666 * width, y: 0.99058 * height),
            control2: CGPoint(x: 0.51381 * width, y: height))
        path.addLine(to: CGPoint(x: 0.42906 * width, y: height))
        path.addCurve(
            to: CGPoint(x: 0.4219 * width, y: 0.97659 * height),
            control1: CGPoint(x: 0.42478 * width, y: height),
            control2: CGPoint(x: 0.4219 * width, y: 0.99067 * height))
        path.addLine(to: CGPoint(x: 0.4219 * width, y: 0.46232 * height))
        path.addCurve(
            to: CGPoint(x: 0.42906 * width, y: 0.43891 * height),
            control1: CGPoint(x: 0.4219 * width, y: 0.44833 * height),
            control2: CGPoint(x: 0.42475 * width, y: 0.43891 * height))
        path.addLine(to: CGPoint(x: 0.58997 * width, y: 0.43891 * height))
        path.addCurve(
            to: CGPoint(x: 0.62214 * width, y: 0.54408 * height),
            control1: CGPoint(x: 0.60964 * width, y: 0.43891 * height),
            control2: CGPoint(x: 0.62214 * width, y: 0.48098 * height))
        path.addLine(to: CGPoint(x: 0.62214 * width, y: 0.61423 * height))
        path.addCurve(
            to: CGPoint(x: 0.61498 * width, y: 0.63764 * height),
            control1: CGPoint(x: 0.62214 * width, y: 0.62822 * height),
            control2: CGPoint(x: 0.61929 * width, y: 0.63764 * height))
        path.addLine(to: CGPoint(x: 0.54884 * width, y: 0.63764 * height))
        path.addCurve(
            to: CGPoint(x: 0.52738 * width, y: 0.5675 * height),
            control1: CGPoint(x: 0.5356 * width, y: 0.63764 * height),
            control2: CGPoint(x: 0.52738 * width, y: 0.60957 * height))
        path.addLine(to: CGPoint(x: 0.52738 * width, y: 0.52076 * height))
        path.addLine(to: CGPoint(x: 0.51666 * width, y: 0.52076 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.72762 * width, y: 0.56759 * height))
        path.addCurve(
            to: CGPoint(x: 0.70616 * width, y: 0.63774 * height),
            control1: CGPoint(x: 0.72762 * width, y: 0.60966 * height),
            control2: CGPoint(x: 0.7194 * width, y: 0.63774 * height))
        path.addLine(to: CGPoint(x: 0.64002 * width, y: 0.63774 * height))
        path.addCurve(
            to: CGPoint(x: 0.63286 * width, y: 0.61432 * height),
            control1: CGPoint(x: 0.63574 * width, y: 0.63774 * height),
            control2: CGPoint(x: 0.63286 * width, y: 0.62841 * height))
        path.addLine(to: CGPoint(x: 0.63286 * width, y: 0.54417 * height))
        path.addCurve(
            to: CGPoint(x: 0.66503 * width, y: 0.439 * height),
            control1: CGPoint(x: 0.63286 * width, y: 0.48107 * height),
            control2: CGPoint(x: 0.64537 * width, y: 0.439 * height))
        path.addLine(to: CGPoint(x: 0.8009 * width, y: 0.439 * height))
        path.addCurve(
            to: CGPoint(x: 0.83308 * width, y: 0.54417 * height),
            control1: CGPoint(x: 0.82057 * width, y: 0.439 * height),
            control2: CGPoint(x: 0.83308 * width, y: 0.48107 * height))
        path.addLine(to: CGPoint(x: 0.83308 * width, y: 0.92985 * height))
        path.addCurve(
            to: CGPoint(x: 0.81162 * width, y: height),
            control1: CGPoint(x: 0.83308 * width, y: 0.97192 * height),
            control2: CGPoint(x: 0.82485 * width, y: height))
        path.addLine(to: CGPoint(x: 0.66501 * width, y: height))
        path.addCurve(
            to: CGPoint(x: 0.63283 * width, y: 0.89482 * height),
            control1: CGPoint(x: 0.64534 * width, y: height),
            control2: CGPoint(x: 0.63283 * width, y: 0.95793 * height))
        path.addLine(to: CGPoint(x: 0.63283 * width, y: 0.74291 * height))
        path.addCurve(
            to: CGPoint(x: 0.65429 * width, y: 0.67276 * height),
            control1: CGPoint(x: 0.63283 * width, y: 0.70084 * height),
            control2: CGPoint(x: 0.64106 * width, y: 0.67276 * height))
        path.addLine(to: CGPoint(x: 0.73831 * width, y: 0.67276 * height))
        path.addLine(to: CGPoint(x: 0.73831 * width, y: 0.52085 * height))
        path.addLine(to: CGPoint(x: 0.7276 * width, y: 0.52085 * height))
        path.addLine(to: CGPoint(x: 0.7276 * width, y: 0.56759 * height))
        path.closeSubpath()
        path.move(to: CGPoint(x: 0.72762 * width, y: 0.91815 * height))
        path.addLine(to: CGPoint(x: 0.73834 * width, y: 0.91815 * height))
        path.addLine(to: CGPoint(x: 0.73834 * width, y: 0.75453 * height))
        path.addLine(to: CGPoint(x: 0.72762 * width, y: 0.75453 * height))
        path.addLine(to: CGPoint(x: 0.72762 * width, y: 0.91815 * height))
        path.closeSubpath()
        return path
    }
}
