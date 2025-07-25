check:
	xcodebuild -project tora.xcodeproj -scheme tora -destination 'platform=iOS Simulator,name=iPhone 16 Pro' build

lint:
	swiftlint lint --strict

run:
	$(eval APP_PATH := $(shell xcodebuild -scheme tora -project tora.xcodeproj -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 16 Pro' -showBuildSettings | grep BUILT_PRODUCTS_DIR | head -1 | cut -d= -f2 | xargs)/tora.app)
	xcodebuild -scheme tora -project tora.xcodeproj -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
	xcrun simctl boot "iPhone 16 Pro" || true
	xcrun simctl install booted $(APP_PATH)
	xcrun simctl launch booted taigaishida.tora-tracker
