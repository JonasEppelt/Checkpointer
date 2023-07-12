REPO_NAME="checkpointer_zipforship"
git push
git clone git@gitlab.etp.kit.edu:jeppelt/checkpointer.git $REPO_NAME
tar -czf checkpointer.tar $REPO_NAME
rm -rf $REPO_NAME
