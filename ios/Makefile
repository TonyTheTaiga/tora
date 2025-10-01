API_BASE_URL ?= http://localhost:8080

check:
	API_BASE_URL=$(API_BASE_URL) xcodebuild -project tora.xcodeproj -scheme tora -destination 'platform=iOS Simulator,name=iPhone 16 Pro' build

lint:
	swiftlint lint --strict

run:
	$(eval APP_PATH := $(shell API_BASE_URL=$(API_BASE_URL) xcodebuild -scheme tora -project tora.xcodeproj -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 16 Pro' -showBuildSettings | grep BUILT_PRODUCTS_DIR | head -1 | cut -d= -f2 | xargs)/tora.app)
	API_BASE_URL=$(API_BASE_URL) xcodebuild -scheme tora -project tora.xcodeproj -configuration Debug -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
	xcrun simctl boot "iPhone 16 Pro" || true
	xcrun simctl install booted $(APP_PATH)
	xcrun simctl launch booted taigaishida.tora-tracker

release:
	@if [ -z "$(RELEASE_API_BASE_URL)" ]; then \
		echo "RELEASE_API_BASE_URL must be set" ; \
		exit 1 ; \
	fi
	API_BASE_URL=$(RELEASE_API_BASE_URL) xcodebuild -project tora.xcodeproj -scheme tora -configuration Release clean archive -archivePath ./build/tora.xcarchive
